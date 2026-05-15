import numpy as np
from exp_article_test.config import UAVConfig

class UAV(object):
	def __init__(self, team : str, id : str):
		self.team = team      # 红蓝队 "Red", "Blue"
		self.id = id          # 编号"0"-"9"，字符串类型

		self.pos = None       # 位置，三维ndarray，单位：米
		self.airspeed = None  # 速度标量，一维，单位：米/秒
		self.vel = None       # 速度矢量，三维ndarray，单位：米/秒
		self.acc = None       # 加速度矢量，三维ndarray，单位：米/秒2
		self.theta = None     # 俯仰角，弧度
		self.phi = None       # 滚转角，弧度
		self.psi = None       # 偏航角，弧度
		self.blood = None     # 剩余血量，一维float，0.0-1.0
		self.alive = True     # 存活状态，True or False
		self.info = "alive"   # 存活信息，字符串类型,"alive","hit","out","crash"

		# 用于记录无人机存活期间的全部历史状态
		self.record_t = []          # 本机存活的全部时间，一般为[0, 0.1, 0.2,...]
		self.record_pos = []        # 位置，x=record_pos[i][0], y=record_pos[i][1], z=record_pos[i][2]为第i/10秒无人机位置
		self.record_airspeed = []   # 速度标量
		self.record_theta = []      # 俯仰角，弧度
		self.record_phi = []        # 滚转角，弧度
		self.record_psi = []        # 偏航角，弧度
		self.record_blood = []      # 血量
		self.record_target = [0]    # *未实际使用的功能，用于记录无人机攻击目标，需注意保存 int编号还是str编号

		self.config = UAVConfig # 读取config.py文件中的配置
		self.airspeed_min = self.config['airspeed_min']   # 最小速度约束
		self.airspeed_max = self.config['airspeed_max']   # 最大速度约束
		self.t = 0.0 # 本机时间
		self.dt = 0.1 # 仿真间隔

	def reset(self, pos, airspeed, psi, theta=0., phi=0.):
		self.reset_status_params() # 清空记录表，并重置生存参数
		self.pos = pos             #
		self.airspeed = airspeed   #
		self.theta = theta
		self.psi = psi
		self.phi = phi
		Cthe = np.cos(self.theta)
		Sthe = np.sin(self.theta)
		Cpsi = np.cos(self.psi)
		Spsi = np.sin(self.psi)
		self.vel = np.array([self.airspeed * Cthe * Cpsi, self.airspeed * Cthe * Spsi, -self.airspeed * Sthe])
		self.acc = np.array([0., 0., 0.])
		self.record_status()

	def reset_status_params(self):
		self.alive = True
		self.blood = 1.0
		self.info = "alive"
		self.t = 0.0
		self.record_t = []
		self.record_pos = []
		self.record_airspeed = []
		self.record_theta = []
		self.record_psi = []
		self.record_phi = []
		self.record_blood = []
		self.record_target = []

	def record_status(self):
		self.record_t.append(self.t)
		self.record_pos.append(self.pos)
		self.record_airspeed.append(self.airspeed)
		self.record_theta.append(self.theta)
		self.record_psi.append(self.psi)
		self.record_phi.append(self.phi)
		self.record_blood.append(self.blood)

	def step(self, action):
		self.t += self.dt
		nx = action[0]
		nn = action[1]
		phi = action[2]
		self.phi = phi
		Cthe = np.cos(self.theta)
		Sthe = np.sin(self.theta)
		Cpsi = np.cos(self.psi)
		Spsi = np.sin(self.psi)
		Cphi = np.cos(self.phi)
		Sphi = np.sin(self.phi)
		speed = self.airspeed + 1e-9
		self.acc = np.array([self.airspeed * Cthe * Cpsi, self.airspeed * Cthe * Spsi, -self.airspeed * Sthe])
		self.pos = np.array(self.pos) + np.array(self.vel) * self.dt
		self.airspeed += 9.8 * (nx - Sthe) * self.dt
		self.airspeed = np.clip(self.airspeed, self.airspeed_min, self.airspeed_max)
		self.theta += 9.8 * (nn * Cphi - Cthe) / (speed + 1e-6) * self.dt
		self.psi += 9.8 * nn * Sphi / (speed * Cthe + 1e-6) * self.dt
		self.clip_params()
		self.update_velocity()

	def set_target(self, target_id):
		self.record_target.append(target_id)

	def update_velocity(self):
		Cthe = np.cos(self.theta)
		Sthe = np.sin(self.theta)
		Cpsi = np.cos(self.psi)
		Spsi = np.sin(self.psi)
		self.vel = np.array([self.airspeed * Cthe * Cpsi, self.airspeed * Cthe * Spsi, -self.airspeed * Sthe])
		self.acc = (self.vel - self.acc) / (self.dt + 1e-6)

	def clip_params(self):
		if self.theta > np.pi or self.theta <= -np.pi:
			while self.theta <= -np.pi:
				self.theta += np.pi * 2
			while self.theta > np.pi:
				self.theta -= np.pi * 2
		if self.psi > np.pi or self.psi <= -np.pi:
			while self.psi <= -np.pi:
				self.psi += np.pi * 2.0
			while self.psi > np.pi:
				self.psi -= np.pi * 2.0

	def destroy(self, info):
		self.info = info
		self.alive = False
		self.blood = 0.0
		self.record_status()
