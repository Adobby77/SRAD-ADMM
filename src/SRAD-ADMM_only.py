import numpy as np
import threading
import queue
import time
import matplotlib.pyplot as plt
import logging
import os

# --- 1. 환경 및 로깅 설정 ---
def get_compute_delay(agent_id):
    """
    각 에이전트의 연산 속도를 결정합니다. 
    0번 에이전트를 느린 노드(Straggler)로 설정하여 비동기성을 테스트합니다.
    """
    if agent_id == 0:
        return np.random.uniform(0.5, 1.0) # Straggler
    return np.random.uniform(0.01, 0.05)   # Normal agents

def setup_logger(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename): os.remove(filename)
    header = 'Time,AgentID,K_Local,K_Read,State,Event,PeerID,Details,LocalObj,GlobalObj,ConsensusErr\n'
    with open(filename, 'w') as f: f.write(header)
    l = logging.getLogger('SRAD_Only_Logger')
    l.setLevel(logging.INFO)
    fh = logging.FileHandler(filename)
    fh.setFormatter(logging.Formatter('%(message)s'))
    l.addHandler(fh)
    return l

# 로그 저장 경로
srad_csv = setup_logger('../results/srad_only_run.csv')

# --- 2. SRAD-ADMM 에이전트 클래스 ---
class SRADAgent(threading.Thread):
    def __init__(self, agent_id, N_total, N_min, rho, local_data, global_data, all_queues):
        super().__init__()
        self.id, self.N_total, self.N_min, self.rho = agent_id, N_total, N_min, rho
        self.all_queues = all_queues
        self.msg_queue = all_queues[agent_id]
        self.A_i, self.b_i = local_data
        self.A_g, self.b_g = global_data
        dim = self.A_i.shape[1]
        
        # 초기 변수 설정
        self.x_curr = np.random.uniform(-0.5, 0.5, dim)
        self.y_curr = -rho * self.x_curr
        self.z_curr = np.zeros(dim); self.z_next = np.zeros(dim)
        self.s_curr = np.zeros(dim); self.s_next = np.zeros(dim)
        self.x_pre = self.x_curr.copy(); self.y_pre = self.y_curr.copy(); self.s_pre = self.s_curr.copy()
        
        self.k_pre = 0; self.k_read = 0; self.j_read = 0; self.j = 0; self.state = 0; self.contributions = 0
        self.i_read = agent_id; self.N_count = 0; self.start_time = time.time()
        self.history = []
        
        # t=0 초기 오차 기록
        init_obj = np.linalg.norm(self.A_g @ self.z_curr - self.b_g)**2
        self.history.append((0, init_obj))

    def log_data(self, event, peer="-", details="-"):
        l_obj = np.linalg.norm(self.A_i @ self.z_curr - self.b_i)**2
        g_obj = np.linalg.norm(self.A_g @ self.z_curr - self.b_g)**2
        c_err = np.linalg.norm(self.x_curr - self.z_curr)
        srad_csv.info(f"{time.time()-self.start_time:.4f},{self.id},{self.k_pre},{self.k_read},{self.state},{event},{peer},{details},{l_obj:.6f},{g_obj:.6f},{c_err:.6f}")

    def run(self):
        while self.k_pre < 100:
            if self.msg_queue.empty():
                # Algorithm 3: Local Process
                if self.state == 0: # x-update
                    L = 2*self.A_i.T@self.A_i + self.rho*np.eye(self.A_i.shape[1])
                    R = 2*self.A_i.T@self.b_i + self.rho*self.z_curr - self.y_curr
                    self.x_next = np.linalg.solve(L, R)
                elif self.state == 1: # y-update
                    self.y_next = self.y_curr + self.rho*(self.x_next-self.z_curr) if self.contributions > 0 else self.rho*(self.x_next-self.x_curr)
                elif self.state == 2: self.s_next = self.x_next + (self.y_next / self.rho)
                elif self.state == 3: # z-aggregation
                    self.j = 1 if (self.k_pre+1) > self.k_read else self.j_read + 1
                    if self.contributions == 0:
                        self.N_count += 1
                        self.z_next = (1/self.N_count)*((self.N_count-1)*self.z_next + self.s_next)
                    else:
                        self.z_next = self.z_next + (1/self.N_count)*(self.s_next - self.s_curr)
                elif self.state == 4: # broadcast
                    for q in self.all_queues.values():
                        if q != self.msg_queue: q.put((self.z_next, self.id, self.j, self.k_pre+1, self.N_count))
                    self.s_curr, self.y_curr, self.x_curr = self.s_next.copy(), self.y_next.copy(), self.x_next.copy()
                    self.j_read, self.k_read, self.contributions = self.j, self.k_pre+1, self.contributions + 1
                
                if self.state >= 4:
                    if self.j_read >= self.N_min:
                        self.z_curr = self.z_next.copy()
                        g_obj = np.linalg.norm(self.A_g @ self.z_curr - self.b_g)**2
                        self.history.append((time.time() - self.start_time, g_obj))
                        self.log_data("SUCCESS")
                        self.k_pre += 1; self.state = 0; self.contributions = 0
                    else: self.state = 3
                else: self.state += 1
                time.sleep(get_compute_delay(self.id))
            else:
                # Algorithm 4: Messaging
                msg = self.msg_queue.get()
                z_un, i_un, j_un, k_un, N_un = msg
                if not (i_un < self.i_read and j_un == self.j_read and k_un == self.k_read):
                    if k_un > self.k_read: self.z_curr = z_un.copy(); self.k_pre = k_un - 1
                    self.i_read, self.j_read, self.N_count, self.k_read, self.z_next = i_un, j_un, N_un, k_un, z_un.copy()
                    if (self.k_pre + 1) != self.k_read: self.state = 0

# --- 3. 메인 실행 ---
if __name__ == "__main__":
    N_total, N_min, rho = 30, 20, 2.0
    A_f = np.random.randn(1500, 10); x_t = np.random.randn(10)
    b_f = A_f @ x_t + np.random.randn(1500) * 0.1
    
    all_q = {i: queue.Queue() for i in range(N_total)}
    agents = [SRADAgent(i, N_total, N_min, rho, (A_f[i*50:i*50+50], b_f[i*50:i*50+50]), (A_f, b_f), all_q) for i in range(N_total)]
    
    print(">>> SRAD-ADMM 전용 모드 시작...")
    for a in agents: a.start()
    for a in agents: a.join()

    # 결과 플롯
    plt.figure(figsize=(10, 6))
    srad_all = []
    for a in agents: srad_all.extend(a.history)
    srad_all.sort()
    t_s, o_s = zip(*srad_all)
    plt.plot(t_s, o_s, 'r-', label=f'SRAD-ADMM (N_min={N_min})')
    plt.yscale('log'); plt.xlabel('Time (s)'); plt.ylabel('Global Objective F(z)')
    plt.title('SRAD-ADMM Convergence History')
    plt.legend(); plt.grid(True); plt.savefig('../results/SRAD-ADMM(only).png')
    print("SRAD-ADMM finished. Plot saved to ../results/SRAD-ADMM(only).png")