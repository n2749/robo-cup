# Soccer Simulation Assignment Requirements

## Assignment Tasks

1. **Write BDI architecture for each player in form of pseudocode**
2. **Write a code in Python or other to implement self-evolving agent learning at every round of play by applying Reinforcement learning**
3. **Demonstrate by implementing a code that teams are playing better by showing efficiency of wins and show results**
4. **Prepare a presentation and defend during the lecture**

---

## Core System Requirements

### Individual Robot World Model

Each robot must maintain its own world model updated via `UPDATEWORLDMODEL()` procedure:

```
Procedure UPDATEWORLDMODEL(robot_position, robot_angle, ball_pos, op_pos, current)
- UPDATELOCALIZATION(robot_posn, robot_ang)
  - wm_position = robot_position
  - wm_heading = robot_angle
- UPDATEVISION(ball_pos, op_pos, current)
- UPDATESHAREDINFORMATION(current)
- UPDATETIME(current)
```

The procedure that updates the world model to account for new localization information simply copies the localization information into `wm_position` and `wm_heading`.

### Soccer Field Physics

- **Soccer field is 2D rectangular, 100 x 65 meters**
- **Center of the soccer field is set to (0,0), Y goes up, X goes right**
- **Players and ball are treated as circles**
- **Movements simulated stepwise for every 50 milliseconds (24 steps/sec)**

**Movement Physics:**
```
At each simulation step, movement of each player is calculated as:
- P1 = P0 + V0;
- V1 = V0 + A0;
- A1 = FORCE * K1 - V0 * K2;

Movement of the ball is calculated as:
- P1 = P0 + V0;
- V1 = V0 + A0;
- If (kicked by a player)
  {
    A1 = KICKFORCE * K1;
    V1 = 0;
  }
  else A1 = -FRICTIONFACTOR * V0;
```

**Collisions:**
- When more than two players overlap, all of them are moved back until they do not overlap
- Then their velocities are multiplied by -0.1

### Soccer Rules Implementation

**Match Cycle:**
- Once the server starts, it enters a 4-period match cycle repeatedly until turned off:
  - **Pre Game:** The referee is not activated. No score is recorded.
  - **First Half:** The game starts. The referee is activated to enforce the soccer rule.
  - **Half Time:** The referee is deactivated again.
  - **Second Half:** The game restarts. The referee is reactivated to enforce the soccer rule.

**Kick-off Rules:**
- A kick-off is a way of starting or restarting play at the start of the match, or after a goal has been scored or at the start of the second half of the match
- When the kick-off mode is on, the opponents of the team taking the kick-off are at least 9 meters from the ball until the ball is touched by an opponent player

**Other implemented soccer rules:**
- Goal kick
- Corner kick  
- Throw in
- Offside (can be turned off by command-line switch)

### Player Roles, Responsibilities and Strategies

| Role | Responsibility | Strategies |
|------|---------------|------------|
| Free-kick-taker | take-corner | pass |
| Attacker | get-ball, score-goal | pass, tackle, shoot |
| Defender | defend-goal, clear-ball | block, mark, pass |
| Player | position, move-ball | go-zone, pass |

### Corner Kick Example
- Must handle set-piece scenarios with coordinated player movement
- Show tactical positioning and execution (as shown in diagram)

---

## Advanced World Model Requirements

### Ball Tracking System

```
Procedure GETBALLLOCATION(τcurrent, robot_id)
ball = NIL
best_confidence = σthreshold
for i = 1 ... n
    if i ≠ robot_id
        if isVALID(i, τcurrent)
            if σsum_ball_i < best_confidence
                best_confidence = σsum_ball_i
                ball = sum_ball_i
return ball
```

**Validation Function:**
```
Procedure isVALID(i, τcurrent)
if i < 0 or i > n
    return FALSE
if τcurrent - τsum_ball_i > τthreshold
    return FALSE
if σsum_ball_i > σthreshold
    return FALSE
if sum_sawball_i ≠ false
    return FALSE
return TRUE
```

### Temporal Updates

**Because the robot soccer environment is dynamic, we expect objects to move over time from where the robot last observed them.**

```
Procedure UPDATETIME(τcurrent)
If any object has not been updated this time period, add some error to its standard deviation.
    if τsum_ball ≠ τcurrent
        σsum_ball = σsum_ball + SMALL_ERROR
    for i = 1 to m
        σsum_opponent_i = σsum_opponent_i + SMALL_ERROR
```

### Shared Information System

**If the ball has not been observed by the robot for a period of time greater than threshold, the best available ball location is requested from the shared world model, using the GETBALLLOCATION function**

```
Procedure UPDATESHAREDINFORMATION(τcurrent)
If the ball has not been seen in a long time, request its location from the shared world model.
    If τcurrent - τsum_ball > τthreshold
        shared_ball = GETBALLLOCATION(τcurrent, robot_id)
        if shared_ball ≠ NIL
            wm_ball = MERGE(wm_ball, shared_ball)
            τsum_ball = τcurrent
    Get teammate location from the shared world model
        for i = 1 to n
            wm_teammate_i = GETTEAMMATELOCATION(i)
```

### Vision System Simulation

**Both the ball position, wm_ball, and the opponent position vector, wm_opponent, are updated from the information returned by the vision module, as described below:**

```
Procedure UPDATEVISION(ball_pos, op_pos, τcurrent)
Update the ball position.
    if ball_pos ≠ NIL
        μglobal = μwm_position + ROTATE(μball_pos, μwm_heading)
        σglobal = SMALL_ERROR
        MERGE(wm_ball, {μglobal, σglobal})
        τwm_ball = τcurrent
    Update the opponent vector.
        for i = 1 to SIZE(op_pos)
            μglobal = μwm_position + ROTATE(μop_pos_i, μwm_heading)
            j = arg min_k ||μwm_opponent_k - μglobal||
            dist = ||μwm_opponent_j - μglobal||
            if (dist < OP_THRESHOLD)
                σglobal = SMALL_ERROR
                MERGE({μglobal, σglobal}, wm_opponent_j)
                τwm_opponent_j = τcurrent
            else
                j = arg max_k(τcurrent - τwm_opponent_k)
                σglobal = SMALL_ERROR
                MERGE({μglobal, σglobal}, wm_opponent_j)
                τwm_opponent_j = τcurrent
```

---

## Implementation Checklist

### Core Requirements
- [ ] BDI architecture pseudocode documentation
- [ ] Reinforcement learning implementation for self-evolving agents
- [ ] Performance metrics showing team improvement
- [ ] Presentation preparation

### World Model Enhancements
- [ ] Confidence-based ball tracking with `GETBALLLOCATION`
- [ ] Temporal uncertainty growth (`UPDATETIME`)
- [ ] Shared information system between teammates
- [ ] Vision system with coordinate transformations
- [ ] Validation system for observations

### Physics & Rules
- [ ] 100x65m field with proper coordinate system
- [ ] 50ms timestep simulation (20Hz)
- [ ] Force-based movement with friction
- [ ] Collision detection and resolution
- [ ] Soccer rules: kick-off, corner kick, goal kick, throw-in
- [ ] Referee system with match cycle phases

### Player System
- [ ] Role-based player types (Attacker, Defender, Free-kick-taker, Player)
- [ ] Strategy implementations for each role
- [ ] Set-piece coordination (corner kicks, etc.)

### Performance Demonstration
- [ ] Win efficiency metrics
- [ ] Learning curve visualization
- [ ] Comparative analysis (before/after training)
- [ ] Statistical significance testing

---

## Technical Notes

- **Simulation Rate:** 50ms timesteps (20 steps/second)
- **Field Dimensions:** 100m x 65m rectangular
- **Coordinate System:** Center (0,0), Y-up, X-right
- **Physics:** Force-based movement with friction and collision handling
- **Communication:** Shared world model for teammate coordination
- **Vision:** Limited range with uncertainty and temporal decay
- **Learning:** Reinforcement learning for continuous improvement

## Deliverables

1. **Code Implementation** - Complete Python system with RL agents
2. **BDI Pseudocode** - Formal documentation of architecture
3. **Performance Analysis** - Metrics and results showing improvement
4. **Presentation** - Defense of implementation and results