# 动作空间从零：DoF、正逆运动学、旋转表示

> [01](01-action-space.md) 是结论版。这一篇是**机器人前置**：从"一条 action 里每个数字到底是什么"讲起。
> 没做过机器人的话先看这篇，看完再回 01 会很快。

一句话：读任何 VLA 之前必须能回答的问题是

> **模型输出的那个向量，每一维是什么意思？**

## 1. DoF：有多少个独立可控的量

一个 6 关节机械臂就是 **6 DoF**，加一个夹爪，动作向量可以是 7 维。

机械臂的解剖只有两个词：

```text
Base
 ↓
Joint 1     ← 关节：可以转 / 可以移动的地方
 ↓
Link        ← 连杆：两个关节之间的刚性结构，不能弯
 ↓
Joint 2
 ↓
...
 ↓
Joint 6
 ↓
Gripper     ← end effector（也可以是吸盘、灵巧手）
```

对人的手臂：肩 → 上臂 → 肘 → 前臂 → 腕 → 手。一样的结构。

关节角通常记成

$$q=[q_1,q_2,\dots,q_6],\qquad \text{单位 rad}$$

比如 $q=[0.2,-0.8,1.1,0.4,-0.3,0.7]$。

$$\boxed{\text{6 个关节角一确定，整条机械臂的形状就确定了}}$$

这句话是后面 FK 的全部依据。

## 2. 第一种 action space：关节空间

直接命令每个关节。可以给目标角：

$$a_t=[0.21,-0.79,1.12,0.40,-0.31,0.69]$$

也可以给增量：

$$\Delta q=[+0.01,+0.01,+0.02,0,-0.01,-0.01],\qquad q_{t+1}=q_t+\Delta q$$

⚠️ **两个都是关节空间。** 「关节 vs 笛卡尔」和「绝对 vs 增量」是**两个独立的维度**，刚开始最容易搅在一起。

## 3. 第二种：笛卡尔 / 末端空间

人抓杯子的时候脑子里不会想"肩关节转 12 度、肘关节转 -17 度"，想的是"我的手移动到杯子旁边"。所以更自然的方式是直接控制**末端执行器（end effector, EE）在哪**。

位置好说：

$$p=[x,y,z]=[0.5,0.2,0.3]$$

（基座前方 0.5 m、左 0.2 m、高 0.3 m。）

但位置不够。**手可以在桌上同一个点，手掌朝上、朝下、朝左**，这是不同的状态。所以还要 orientation：

$$\text{Pose} = \text{Position} + \text{Orientation} = [x,y,z] + [r_x,r_y,r_z]$$

因为 $x,y,z$ 来自三维笛卡尔坐标系，这种表示叫 **Cartesian space**，也叫 **task space** 或 **end-effector space**，入门阶段可以当同一个东西。

## 4. FK 和 IK：两个空间怎么互转

### 正运动学 FK

关节角确定了整条臂的形状，所以能算出手在哪：

$$q \longrightarrow [x,y,z,R]$$

二维版最好懂：

```text
shoulder
   O
    \  L1
     O elbow
      \  L2
       X hand
```

已知 $q_1,q_2$ 和杆长 $L_1,L_2$，就能算出手的位置 $(x,y)$。这就是 FK。

### 逆运动学 IK

反过来：VLA 说"手去 $[0.5,0.2,0.3]$"，但驱动的是电机，电机不认识 $x=0.5$。所以要解

$$[x,y,z,R] \longrightarrow [q_1,\dots,q_6]$$

这就是 IK。完整链条：

```text
VLA
 ↓  目标 EE pose [x,y,z,R]
IK
 ↓  目标关节角 [q1..q6]
motor controller
 ↓
机器人运动
```

$$\boxed{\text{高层：VLA 决定"手怎么动"；低层：controller 决定"电机怎么动"}}$$

### 那为什么不干脆让 VLA 直接输出关节

可以，很多数据集就是关节动作。但有一个致命问题：**不同机器人的关节完全对不上。**

```text
机器人 A:  [q1..q6]
机器人 B:  [q1..q7]
机器人 C:  左臂 7 + 右臂 7 + 腰 3 + 头 2
```

action space 结构都不一样。而它们**都有**"左手在哪、右手在哪、夹爪开合"这些概念。所以末端空间更容易做 [cross-embodiment](../06-vla/02-gen1-ar-vla.md#rt-x-的-cross-embodiment-是粗对齐)。

## 5. 控制量还分几层

即使都在关节空间，控制的物理量也可以不同：

```text
End-effector pose
        ↓
Joint position     ← q_target，"转到这里"
        ↓
Joint velocity     ← q̇，"以多快转"
        ↓
Joint torque       ← τ，"施加多大力矩"
        ↓
Motor current
```

VLA 一般在上面两层（EE pose / EE delta，或 joint position / joint delta），**不会直接预测电机电流**。

## 6. 四种组合

「关节 vs 笛卡尔」× 「绝对 vs 增量」：

| 表示 | 模型预测什么 |
|---|---|
| Joint absolute | 每个关节的目标角度 |
| Joint delta | 每个关节相对转多少 |
| Cartesian absolute | 手去哪个空间坐标 |
| Cartesian delta | 手相对当前移动多少 |

具体数字：当前手在 $[0.50,0.20,0.30]$，

- Cartesian absolute 输出 $[0.52,0.18,0.31]$，意思"直接去这个坐标"
- Cartesian delta 输出 $[+0.02,-0.02,+0.01]$，$p_{t+1}=p_t+\Delta p$，结果一样

## 7. Orientation 到底是什么

先不管欧拉角和四元数。最本质的一句话：

> **刚体自己也有一套坐标轴，orientation 描述的是这套局部坐标轴在世界坐标系里分别指向哪。**

```text
   手的局部坐标            世界坐标
     z_hand                z_world
       ↑                     ↑
       |                     |
       O ──→ x_hand          O ──→ x_world
      /                     /
   y_hand                y_world
```

如果 $x_{\text{hand}}$ 在世界里就是 $[1,0,0]$、$y_{\text{hand}}$ 是 $[0,1,0]$、$z_{\text{hand}}$ 是 $[0,0,1]$，那两套坐标系完全一致：

$$R=\begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}$$

如果手绕世界 $z$ 轴转了 90°，那么 $x_{\text{hand}}$ 指向世界的 $y$、$y_{\text{hand}}$ 指向世界的 $-x$：

$$R=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}$$

这就是 **rotation matrix**，$R\in\mathbb R^{3\times3}$，9 个数字。

### 但旋转只有 3 个自由度

因为这 9 个数字被约束死了：

$$R^\top R=I,\qquad \det(R)=1$$

真正独立的信息只有 3 个。这也很合理：绕 x、绕 y、绕 z，三种独立旋转。

$$\boxed{9\text{ 个数字，但 }3\text{ 个自由度}}$$

## 8. 四种旋转表示

### 欧拉角：3 个数

$$[\alpha,\beta,\gamma] = [\text{roll},\text{pitch},\text{yaw}]$$

拿飞机记：

| | 绕什么转 | 现象 |
|---|---|---|
| **Roll** | 前后轴 | 左翼下沉、右翼抬起 |
| **Pitch** | 左右轴 | 机头抬起 / 压下 |
| **Yaw** | 竖直轴 | 机头向左 / 向右 |

于是有了最经典的 **6D pose**：$[x,y,z,\text{roll},\text{pitch},\text{yaw}]$。

欧拉角有两个问题：

**万向锁（gimbal lock）**：某些姿态下两个旋转轴重合，三个自由度里丢掉一个。比如 pitch 到 $90^\circ$ 附近时，roll 和 yaw 的效果变得无法区分。

**不连续**：$179^\circ$ 和 $-179^\circ$ 在空间里只差 $2^\circ$，但数值上差 $358$。对神经网络非常不友好 —— 它会以为这两个 target 差了十万八千里。

### 四元数：4 个数

$$q_{\text{rot}}=[w,q_x,q_y,q_z],\qquad w^2+q_x^2+q_y^2+q_z^2=1$$

（⚠️ 这个 $q$ 和关节角的 $q$ 只是符号撞了，完全不是一回事。）

有 4 个数但归一化约束掉一个，还是 3 DoF。直觉是轴角：绕单位轴 $\mathbf u$ 转 $\theta$，

$$q=[\cos(\theta/2),\ u_x\sin(\theta/2),\ u_y\sin(\theta/2),\ u_z\sin(\theta/2)]$$

算一个：绕 $z$ 轴转 $90^\circ$，$\mathbf u=[0,0,1]$，

$$q=[\cos45^\circ,0,0,\sin45^\circ]\approx[0.707,0,0,0.707]$$

四元数也不完美：**双重覆盖**，$q$ 和 $-q$ 表示同一个旋转。$[0.707,0,0,0.707]$ 和 $[-0.707,0,0,-0.707]$ 姿态完全相同，直接回归也会有不连续。

### 6D 旋转表示：现在回归的主流

⚠️ **这个 6D 不是 $xyz+rpy$ 那个 6D pose**，是完全不同的概念，极容易混。

思路：不预测欧拉角、不预测四元数，**预测旋转矩阵的前两列**。

$$R=[r_1,r_2,r_3] \quad\Longrightarrow\quad \text{只预测 }r_1,r_2 \quad\Longrightarrow\quad 3+3=6$$

第三列用 Gram-Schmidt 现场重建：

$$b_1=\frac{a_1}{\|a_1\|},\qquad \tilde b_2=a_2-(b_1^\top a_2)b_1,\qquad b_2=\frac{\tilde b_2}{\|\tilde b_2\|},\qquad b_3=b_1\times b_2$$

$$R=[b_1,b_2,b_3]$$

于是任意一个 6D 向量都能还原成合法旋转矩阵，而且这个表示**连续**，适合回归。深层原因见 [01 第 2 节](01-action-space.md#2-姿态怎么表示)：$SO(3)$ 到 $\le4$ 维实空间不存在连续双射。

### 汇总

| 表示 | 数字个数 | 实际 DoF | 特点 |
|---|---:|---:|---|
| 欧拉角 | 3 | 3 | 直观，但万向锁 + 不连续 |
| 四元数 | 4 | 3 | 常用，但双重覆盖 |
| 旋转矩阵 | 9 | 3 | 最直接，冗余大 |
| 6D 表示 | 6 | 3 | 回归友好，现在的主流 |

## 9. 卡点：维度不等于自由度

$$\boxed{\text{tensor dimension}\ \neq\ \text{physical DoF}}$$

**看到论文写 "action dimension = 7"，几乎等于没有信息。** 它可能是：

| 情况 | 含义 |
|---|---|
| A | $[x,y,z,\text{roll},\text{pitch},\text{yaw},\text{gripper}]$ = 3 位置 + 3 欧拉 + 1 夹爪 |
| B | $[x,y,z,q_w,q_x,q_y,q_z]$ = 3 位置 + 4 四元数，**连夹爪都没有** |
| C | $[q_1,\dots,q_7]$ = 七个关节角 |

三种完全不是一回事。

反过来，**同一台 6-DoF 机械臂 + 夹爪**，换个旋转表示维度就变：

| 旋转表示 | 动作维度 |
|---|---|
| 欧拉角 | $3+3+1=7$ |
| 四元数 | $3+4+1=8$ |
| 6D 旋转 | $3+6+1=10$ |

底层物理自由度都是 6 EE + 1 gripper，**没有任何变化**。所以看到 "10-dimensional end-effector action" 时，一个非常可能的定义就是 3 位置 + 6D 旋转 + 1 夹爪。

这条以后读 G0.5 的 **27D unified action space** 时会直接用到：27 不代表"这机器人有 27 个自由度"，要逐维看它在统一哪些本体的控制量。

## 10. 增量旋转有左乘右乘之分

位置的增量是加法，旋转的不是：

$$R_{t+1}=\Delta R_t R_t \qquad\text{或}\qquad R_{t+1}=R_t\Delta R_t$$

左乘右乘对应旋转是在**哪个坐标系**下定义的（世界系还是工具系）。看数据集时这属于必须查清楚的 convention。

## 11. state 和 action 别混

```text
state  s_t = [x_t, y_t, z_t, R_t, g_t]        ← 我现在在哪
action a_t = [Δx, Δy, Δz, ΔR, g_{t+1}]        ← 我下一步怎么动
```

state 就是 proprioception（本体感知），是**输入**；action 是**输出**。

## 12. 完整链条

```text
camera image I_t  +  proprioception s_t  +  instruction l
                          ↓
                         VLA
                          ↓
              Cartesian delta action
              [Δposition, Δrotation, gripper]
                          ↓
                  更新目标 EE pose
                          ↓
                    IK / controller
                          ↓
                      joint target
                          ↓
                   low-level controller
                          ↓
                        motor
                          ↓
                    机器人真的动了
```

## 读论文时的三层追问

```text
第一层：控制哪个空间？   joint / Cartesian
第二层：预测什么物理量？ position / velocity / torque
第三层：绝对还是增量？   absolute / delta
```

三个都回答完，才算看懂了一个 VLA 的 action。最常见的组合是 `Cartesian + position + delta`，也就是

$$[\Delta x,\Delta y,\Delta z,\Delta r_x,\Delta r_y,\Delta r_z,g]$$

## 自测

**1.** joint / link 分别是什么？为什么说关节角一确定形状就确定了？

> **答：** **joint** 是可以转 / 移动的地方，**link** 是两个关节之间不能弯的刚性结构。因为 link 是刚性的，所以只要所有关节角给定，整条链的几何形状唯一确定 —— 这就是 FK 能算出手在哪的依据。

**2.** 「关节空间 vs 笛卡尔空间」和「绝对 vs 增量」是同一个问题吗？

> **答：** **不是，是两个独立的维度**，四种组合都存在。$\Delta q$ 是关节空间的增量控制，$[x,y,z]$ 目标是笛卡尔的绝对控制。刚入门最容易把这两件事搅在一起。

**3.** FK 和 IK 分别是什么？VLA 为什么常输出末端空间？

> **答：** **FK**：$q\to[x,y,z,R]$，关节角算末端位姿。**IK**：$[x,y,z,R]\to q$，末端位姿反解关节角。
> VLA 输出末端空间是因为不同机器人的关节结构完全对不上（6 关节 / 7 关节 / 双臂 + 腰 + 头），但都有"手在哪、夹爪开合"这个共同概念，所以末端空间更容易 cross-embodiment。

**4.** 控制量从高到低有哪几层？VLA 在哪一层？

> **答：** EE pose → joint position → joint velocity → joint torque → motor current。
> VLA 在上面两层（EE pose/delta 或 joint position/delta），不会直接预测电机电流。

**5.** ⭐ orientation 最本质的定义是什么？

> **答：** **刚体自己的三个局部坐标轴，在世界坐标系里分别指向哪。** 把这三个方向向量并成一列就是旋转矩阵 $R$。
> 比如绕世界 $z$ 轴转 90°：$x_{\text{hand}}$ 指向世界 $y$、$y_{\text{hand}}$ 指向世界 $-x$，于是 $R=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}$。

**6.** 旋转矩阵 9 个数字为什么只有 3 个自由度？

> **答：** 被约束死了：$R^\top R=I$（各列单位正交）且 $\det R=1$（右手系）。剩下的独立信息只有 3 个，对应绕 x/y/z 三种独立旋转。

**7.** 欧拉角的两个问题？举一个具体数字。

> **答：** ① **万向锁**：pitch 到 $90^\circ$ 附近时两轴重合，roll 和 yaw 效果不可区分，丢掉一个自由度。
> ② **不连续**：$179^\circ$ 和 $-179^\circ$ 空间上只差 $2^\circ$，数值上差 $358$，网络会以为这两个 target 差得巨大。

**8.** 写出四元数的轴角公式，并算绕 $z$ 轴 90° 的四元数。

> **答：** 绕单位轴 $\mathbf u$ 转 $\theta$：$q=[\cos(\theta/2),u_x\sin(\theta/2),u_y\sin(\theta/2),u_z\sin(\theta/2)]$。
> $\mathbf u=[0,0,1]$、$\theta=90^\circ$：$q=[\cos45^\circ,0,0,\sin45^\circ]\approx[0.707,0,0,0.707]$。
> 四元数的问题是**双重覆盖**：$q$ 和 $-q$ 是同一个旋转。

**9.** 6D 旋转表示怎么还原成旋转矩阵？

> **答：** 预测前两列 $a_1,a_2$，然后 Gram-Schmidt：
> $b_1=\frac{a_1}{\|a_1\|}$，$\tilde b_2=a_2-(b_1^\top a_2)b_1$，$b_2=\frac{\tilde b_2}{\|\tilde b_2\|}$，$b_3=b_1\times b_2$，$R=[b_1,b_2,b_3]$。
> ⚠️ 这个"6D"不是 $xyz+rpy$ 那个 6D pose，两个概念完全不同。

**10.** ⭐ 论文写 "action dimension = 7"，能推出什么？

> **答：** **几乎什么都推不出。** 至少三种可能：$[x,y,z,rpy,g]$（3 位置 + 3 欧拉 + 夹爪）、$[x,y,z,q_w,q_x,q_y,q_z]$（3 位置 + 4 四元数，没有夹爪）、$[q_1..q_7]$（七个关节角）。必须去查 action definition。

**11.** ⭐ 同一台 6-DoF 机械臂 + 夹爪，动作维度可能是几？

> **答：** 欧拉角 $3+3+1=7$；四元数 $3+4+1=8$；6D 旋转 $3+6+1=10$。
> 物理自由度一点没变，变的只是 representation。记：$\boxed{\text{tensor dimension}\neq\text{physical DoF}}$。

**12.** 增量旋转怎么复合？

> **答：** 不是加法。$R_{t+1}=\Delta R_tR_t$ 或 $R_{t+1}=R_t\Delta R_t$，左乘右乘对应旋转在世界系还是工具系下定义，属于必须查清楚的 convention。

**13.** 读一个新 VLA 的 action 时，问哪三层？

> **答：** ① 控制哪个空间（joint / Cartesian）；② 预测什么物理量（position / velocity / torque）；③ 绝对还是增量。
> 最常见的组合是 `Cartesian + position + delta`，即 $[\Delta x,\Delta y,\Delta z,\Delta r_x,\Delta r_y,\Delta r_z,g]$。
