import math, time, heapq, random
from copy import deepcopy

CENTER = 50.0
TIME_LIMIT = 0.9
CPUCT = 1.8
ENEMY_GROWTH = 0
DIST_CACHE = {}
SPEED_CACHE = {}
# ======================
# BASIC MATH
# ======================

def dist(ax, ay, bx, by):
    key = (round(ax,2), round(ay,2), round(bx,2), round(by,2))
    if key not in DIST_CACHE:
        DIST_CACHE[key] = math.hypot(ax - bx, ay - by)
    return DIST_CACHE[key]
def ang(ax, ay, bx, by):
    return math.atan2(by - ay, bx - ax)

LOG1000 = math.log(1000)

def speed(s):
    if s not in SPEED_CACHE:
        SPEED_CACHE[s] = 1.0 + 5.0 * (math.log(max(s,1)) / LOG1000) ** 1.5
    return SPEED_CACHE[s]
def predict_position(p, t, angular_velocity):
    dx = p.x - CENTER
    dy = p.y - CENTER

    r = math.hypot(dx, dy)

    if r < 1e-6 or r > 50:
        return planet_pos(p, t)  # static

    angle0 = math.atan2(dy, dx)
    angle = angle0 + angular_velocity * t

    return CENTER + r * math.cos(angle), CENTER + r * math.sin(angle)

def planet_pos(p, t):
    angle = p.base_angle + ANG_VEL * t
    return (
        CENTER + p.radius * math.cos(angle),
        CENTER + p.radius * math.sin(angle)
    )

def travel_time_to_planet(f, p):

    sp = speed(max(f.ships, 1))

    px, py = planet_pos(p, f.t0)
    dx = px - f.x0
    dy = py - f.y0

    vx = -ANG_VEL * (py - CENTER)
    vy =  ANG_VEL * (px - CENTER)

    a = vx*vx + vy*vy - sp*sp
    b = 2 * (dx*vx + dy*vy)
    c = dx*dx + dy*dy

    disc = b*b - 4*a*c

    if disc < 0 or abs(a) < 1e-6:
        return f.t0 + math.hypot(dx, dy) / sp

    t_hit = (-b - math.sqrt(max(0, disc))) / (2*a)

    if t_hit <= 0:
        return f.t0 + math.hypot(dx, dy) / sp

    return f.t0 + t_hit

def fleet_pos(f, t):
    dt = t - f.t0
    sp = speed(max(f.ships, 1))

    return (
        f.x0 + math.cos(f.ang) * sp * dt,
        f.y0 + math.sin(f.ang) * sp * dt
    )
def intercept_angle(s, t, angular_velocity, send, launch_time=0):
    sp = speed(max(send, 1))

    sx, sy = planet_pos(s, launch_time)
    tx0, ty0 = planet_pos(t, launch_time)

    dx = tx0 - sx
    dy = ty0 - sy

    vx = -angular_velocity * (ty0 - CENTER)
    vy =  angular_velocity * (tx0 - CENTER)

    a = vx*vx + vy*vy - sp*sp
    b = 2 * (dx*vx + dy*vy)
    c = dx*dx + dy*dy

    disc = b*b - 4*a*c

    if disc < 0 or abs(a) < 1e-6:
        return ang(sx, sy, tx0, ty0)

    t_hit = (-b - math.sqrt(max(0, disc))) / (2*a)

    if t_hit <= 0:
        return ang(sx, sy, tx0, ty0)

    tx, ty = planet_pos(t, launch_time + t_hit)
    return ang(sx, sy, tx, ty)

# ======================
# ADAPTIVE HORIZON
# ======================

def get_horizon(planets):
    total = len(planets)

    if total < 12:
        return 14
    elif total < 20:
        return 11
    else:
        return 8
def max_children(planets):
    return int(8 + 50 / (len(planets) + 5))
# ======================
# STATE
# ======================

class P:
    def __init__(self, p):
        self.id, self.owner, self.x, self.y, self.r, self.ships, self.prod = p
        
        # orbit info
        dx = self.x - CENTER
        dy = self.y - CENTER
        self.radius = math.hypot(dx, dy)
        self.base_angle = math.atan2(dy, dx)


class F:
    def __init__(self, owner, x, y, ang, ships, t0, target):
        self.owner = owner
        self.x0 = x
        self.y0 = y
        self.ang = ang
        self.ships = ships
        self.t0 = t0
        self.target = target


# ======================
# NODE
# ======================

class Node:
    def __init__(self, planets, fleets, player, parent=None, action=None, prior=0.0):
        self.future_cache = {}
        self.planets = planets
        self.fleets = fleets
        self.player = player

        self.parent = parent
        self.action = action

        self.children = []
        self.untried = None

        self.visits = 0
        self.value = 0.0

        self.prior = prior


# ======================
# FUTURE RESOLUTION 
# ======================

def future_owner(p, fleets, horizon=6.0):

    net = p.ships
    owner = p.owner

    events = []

    for f in fleets:
        if f.target != p.id:
            continue

        t = travel_time_to_planet(f, p)

        if t <= horizon:
            events.append((t, f))

    events.sort()

    last_t = 0.0

    for t, f in events:

        if owner != -1:
            net += (t - last_t) * p.prod

        last_t = t

        weight = 1.0 / (1.0 + 0.3 * t)

        if f.owner == owner:
            net += f.ships * weight
        else:
            if f.ships * weight > net:
                owner = f.owner
                net = f.ships * weight - net
            
            else:
                net -= f.ships * weight

    return owner, net

def future_owner_cached(node, p, horizon=6.0):
    key = (
        p.id, horizon,
        tuple(sorted(
            (f.owner, f.target, int(f.ships), int(f.t0))
            for f in node.fleets if f.target == p.id
        ))
    )

    if key not in node.future_cache:
        node.future_cache[key] = future_owner(p, node.fleets, horizon)

    return node.future_cache[key]

# ======================
# SIMULATION
# ======================

def simulate(planets, fleets, horizon):

    planets = deepcopy(planets)
    id_map = {p.id: p for p in planets}

    events = []
    # schedule production
    for p in planets:
        if p.owner != -1:
            heapq.heappush(events, (0.0, "prod", p.id))
        # approximate intersection time
        # (constant velocity assumption)
    # schedule fleet arrivals
    for f in fleets:

        if f.target in id_map:
            p = id_map.get(f.target)
        else:
            #  fallback: nearest planet (approximation)
            fx0, fy0 = f.x0, f.y0

            p = min(
                planets,
                key=lambda pl: dist(
                    fx0, fy0,
                    pl.x, pl.y   # already updated
                )
            )
        t = travel_time_to_planet(f, p)

        heapq.heappush(events, (t, "arrive", (p.id, f)))

    current_time = 0.0
    while events:
        t, typ, data = heapq.heappop(events)
        dt = t - current_time

        if dt > 0:
            current_time = t

        # --- UPDATE ALL PLANET POSITIONS TO TIME t ---
        
        if t > horizon:
            break

        if typ == "prod":
            p = id_map[data]
            if p.owner != -1:
                p.ships += p.prod
                heapq.heappush(events, (t + 1, "prod", p.id))

        elif typ == "arrive":
            pid, f = data

            # collect ALL fleets arriving at same time
            arrivals = [(t, f)]
            while events and events[0][0] == t and events[0][1] == "arrive":
                _, _, d2 = heapq.heappop(events)
                arrivals.append((t, d2[1]))

            p = id_map.get(pid)
            if not p:
                continue

            # group forces
            forces = {}
            for _, f in arrivals:
                forces[f.owner] = forces.get(f.owner, 0) + f.ships

            if p.owner != -1:
                forces[p.owner] = forces.get(p.owner, 0) + p.ships

            #  resolve combat (correct rules)
            sorted_forces = sorted(forces.items(), key=lambda x: -x[1])

            if len(sorted_forces) == 1:
                p.owner, p.ships = sorted_forces[0]
            else:
                (o1, s1), (o2, s2) = sorted_forces[:2]
                if s1 > s2:
                    p.owner = o1
                    p.ships = s1 - s2
                elif s2 > s1:
                    p.owner = o2
                    p.ships = s2 - s1
                else:
                    p.owner = -1
                    p.ships = 0

    return planets

# ======================
# VALUE FUNCTION (STABLE + DISCOUNTED)
# ======================

def evaluate(node, planets, fleets, player, horizon):

    score = 0
    decay = 0.85

    for p in planets:

        owner, future = future_owner_cached(node, p)
        prod_value = p.prod * (1 - decay**horizon) / (1 - decay)

        v = future + prod_value

        if owner == player:
            score += v
        else:
            score -= v

    # terminal pressure
    my = sum(1 for p in planets if p.owner == player)
    enemy = sum(1 for p in planets if p.owner == 1-player)

    score += 50 * (my - enemy)

    # snowball
    my_prod = sum(p.prod for p in planets if p.owner == player)
    enemy_prod = sum(p.prod for p in planets if p.owner == 1-player)

    diff = my_prod - enemy_prod
    score += diff * abs(diff) * 0.5
    center_bonus = 0

    for p in planets:
        px, py = planet_pos(p, horizon * 0.5)
        d = dist(px, py, CENTER, CENTER)        
        if p.owner == player:
            center_bonus += 1 / (d + 5)
        elif p.owner == 1 - player:
            center_bonus -= 1 / (d + 5)

    score += 15 * center_bonus

    for f in fleets:
        if f.target < len(planets):
            p = planets[f.target]
            px, py = planet_pos(p, 0)
            d = dist(f.x0, f.y0, px, py)
            f_decay = 1 / (1 + 0.15 * d)
        else:
            f_decay = 0.5

        if f.owner == player:
            score += f.ships * f_decay
        else:
            score -= f.ships * decay
    noise = 0.02 if len(planets) > 20 else 0.005
    score += random.uniform(-noise, noise)
    if enemy == 0:
        return 1.0
    if my == 0:
        return -1.0

    if abs(score) < 200:
        return score / 200
    else:
        return math.tanh(score / 150)

# ======================
# CRITICAL DEFENSE
# ======================

def critical_defense(node, planets, fleets, player):
    acts = []

    for p in planets:
        owner, net = future_owner(p, fleets, 8)
        if p.owner == player and owner != player:
            sources = [s for s in planets if s.owner == player and s.id != p.id]

            if not sources:
                continue

            #  NEW: feasibility filter
            px, py = planet_pos(p, 0)

            candidates = [
                s for s in sources
                if dist(s.x, s.y, px, py) / speed(max(int(s.ships * 0.5), 1)) <= 6
            ]
            
            if not candidates:
                continue

            px, py = planet_pos(p, 0)

            s = max(
            candidates,
            key=lambda x: x.ships / (1 + dist(x.x, x.y, px, py))
            )

            send = int(min(s.ships * 0.5, max(net * 1.2, 10)))
            if send <= 0:
                continue
            px, py = planet_pos(p,0)
            travel = dist(s.x, s.y, px, py) / speed(send)
            if travel > 6:
                continue
            angle = intercept_angle(s, p, ANG_VEL, send, 0.0)
            acts.append((s.id, angle, send, 0.0, p.id))

    return acts


# ======================
# ACTIONS (SAFE + DIVERSE)
# ======================

def gen_actions(planets, player):

    acts = []
    my_prod = sum(p.prod for p in planets if p.owner == player)
    enemy_prod = sum(p.prod for p in planets if p.owner == 1-player)

    my = [p for p in planets if p.owner == player and p.ships > 20]
    enemy = [p for p in planets if p.owner != player]

    for s in my:

        reserve = max(15, s.prod * 3)

        for t in enemy:

            if s.ships <= reserve:
                continue

            send = int((s.ships - reserve) * 0.5)

            if send <= 0 or send > s.ships * 0.7:
                continue
            sx, sy = planet_pos(s, 0)
            tx, ty = planet_pos(t, 0)

            travel = dist(sx, sy, tx, ty) / speed(send)
            if travel > 8:
                continue
            angle = intercept_angle(s, t, ANG_VEL, send)
            acts.append((s.id, angle, send, 0.0, t.id))

        #  chain attack
        for mid in planets:
            if mid.owner == -1:

                send = int((s.ships - reserve) * 0.4)
                if send <= 0:
                    continue

                angle = intercept_angle(s, mid, ANG_VEL, send)
                acts.append((s.id, angle, send, 0.0, mid.id))

        #  fake attack
        if s.ships > 40:
            for t in enemy:
                fake = int(s.ships * 0.1)
                if fake > 5:
                    angle = intercept_angle(s, t, ANG_VEL, fake)
                    acts.append((s.id, angle, fake, 0.0, t.id))

    acts.sort(key=lambda a: -a[2])
    limit = 18 if len(planets) > 15 else 22
    return acts[:limit]
def estimate_capture_time_and_ships(node, source, mid, send):
    travel = dist(source.x, source.y, mid.x, mid.y) / speed(send)

    if mid.owner == -1:
        owner, net = future_owner_cached(node, mid)
        remaining = send - net
    else:
        remaining = send - (mid.ships + mid.prod * travel)

    return travel, max(0, remaining)

def gen_action_sets(node, actions, planets, fleets):

    sets = []
    sets.append([])
    # --- singles ---
    for a in actions[:6]:
        sets.append([a])

    # --- normal combos ---
    for i in range(min(4, len(actions))):
        for j in range(i+1, min(4, len(actions))):
            sets.append([actions[i], actions[j]])

    # --- grouped by target ---
    by_target = {}
    for a in actions:
        by_target.setdefault(a[4], []).append(a)

    id_map = {p.id: p for p in planets}

    for tid, group in by_target.items():

        if len(group) < 2:
            continue

        target = id_map.get(tid)
        if not target:
            continue
        # efficiency sort
        def efficiency(a):
            sid, _, ships, _, _ = a
            s = id_map[sid]
            d = dist(s.x, s.y, target.x, target.y)
            return ships / (d + 5)

        group = sorted(group, key=efficiency, reverse=True)

        main = group[0]

        sx = id_map[main[0]].x
        sy = id_map[main[0]].y

        main_travel = dist(sx, sy, target.x, target.y) / speed(main[2])

        # --- required force ---
        owner, net = future_owner_cached(node, target)
        required = net

        if target.owner != -1:
            required += target.prod * main_travel

        for f in fleets:
            if f.target != tid:
                continue

            t_arrive = travel_time_to_planet(f, target)

            if t_arrive <= main_travel + 1.5:
                if f.owner == target.owner:
                    required += f.ships * 0.9
                else:
                    required -= f.ships * 0.9

        required = max(required, target.ships)
        required *= 1.05 + 0.03 * (len(group) - 1)

        # --- build coordinated ---
        coordinated = []
        total_sent = 0

        for a in group[:5]:

            sid, ang_, ships, _, _ = a

            sx = id_map[sid].x
            sy = id_map[sid].y

            travel = dist(sx, sy, target.x, target.y) / speed(ships)
            delay = main_travel - travel

            if not (-0.8 <= delay <= 1.5):
                continue

            coordinated.append((sid, ang_, ships, max(0.0, delay), tid))
            total_sent += ships

            if total_sent > required * 1.4:
                break

            if total_sent >= required:
                break

        if len(coordinated) >= 2 and required * 0.95 <= total_sent <= required * 1.25:
            sets.append(coordinated)
        # --- grouped by target ---

    # --- TRUE CHAIN ATTACKS ---
    for s in planets:
        if s.owner != node.player or s.ships < 30:
            continue

        for mid in planets:
            if mid.owner != -1:
                continue

            for target in planets:
                if target.owner == node.player:
                    continue

                d1 = dist(s.x, s.y, mid.x, mid.y)
                d2 = dist(mid.x, mid.y, target.x, target.y)

                if d2 > d1 * 2:
                    continue

                send1 = int(s.ships * 0.5)
                if send1 <= mid.ships:
                    continue

                t1 = d1 / speed(send1)
                rem = send1 - mid.ships

                if rem < 5:
                    continue

                delay2 = t1 + 0.3

                angle1 = intercept_angle(s, mid, ANG_VEL, send1)
                angle2 = intercept_angle(mid, target, ANG_VEL, rem)

                chain = [
                    (s.id, angle1, send1, 0.0, mid.id),
                    (mid.id, angle2, rem, delay2, target.id)
                ]

                sets.append(chain)


    # --- MID-FLIGHT MERGE (synchronized attack) ---
    for i in range(len(planets)):
        s1 = planets[i]
        if s1.owner != node.player or s1.ships < 20:
            continue

        for j in range(i+1, len(planets)):
            s2 = planets[j]
            if s2.owner != node.player or s2.ships < 20:
                continue

            for target in planets:
                if target.owner == node.player:
                    continue

                send1 = int(s1.ships * 0.4)
                send2 = int(s2.ships * 0.4)

                if send1 <= 0 or send2 <= 0:
                    continue

                t1 = dist(s1.x, s1.y, target.x, target.y) / speed(send1)
                t2 = dist(s2.x, s2.y, target.x, target.y) / speed(send2)

                # synchronize arrival
                if abs(t1 - t2) > 2.5:
                    continue

                delay1 = max(0.0, t2 - t1)
                delay2 = max(0.0, t1 - t2)

                a1 = intercept_angle(s1, target, ANG_VEL, send1)
                a2 = intercept_angle(s2, target, ANG_VEL, send2)
                merged = [
                    (s1.id, a1, send1, delay1, target.id),
                    (s2.id, a2, send2, delay2, target.id)
                ]

                sets.append(merged)

    def set_score(s):
        return sum(a[2] for a in s) + 10 * len(set(a[4] for a in s))

    sets = sorted(sets, key=set_score, reverse=True)
    return sets[:12]
# ======================
# APPLY
# ======================

def apply(planets, fleets, acts, player):

    planets = deepcopy(planets)
    fleets = list(fleets)

    id_map = {p.id: p for p in planets}

    for pid, a, s, t0, target in acts:

        p = id_map.get(pid)

        if not p or p.owner != player or p.ships < s:
            continue

        p.ships -= s

        #  launch from FUTURE position if delayed
        lx, ly = planet_pos(p, t0)

        fleets.append(F(player, lx, ly, a, s, t0, target))

    return planets, fleets


# ======================
# ENEMY MODEL 
# ======================

def enemy(planets, fleets, player):
    enemy_id = 1 - player
    acts = []

    my_planets = [p for p in planets if p.owner == enemy_id]
    opp_planets = [p for p in planets if p.owner != enemy_id]

    if not my_planets or not opp_planets:
        return acts

    # --- DEFENSE FIRST ---
    for p in my_planets:
        owner, net = future_owner(p, fleets)

        if owner != enemy_id:
            # reinforce
            sources = [s for s in my_planets if s.id != p.id and s.ships > 15]

            if not sources:
                continue
            px, py = planet_pos(p, 0)

            s = min(sources, key=lambda x: dist(x.x, x.y, px, py))
            send = int(s.ships * 0.4)

            if send > 0:
                px, py = planet_pos(p, 0)
                acts.append((s.id, ang(s.x, s.y, px, py), send, 0.0, p.id))

    # --- ATTACK ---
    my_prod = sum(p.prod for p in my_planets)
    opp_prod = sum(p.prod for p in opp_planets)

    if ENEMY_GROWTH > my_prod:
        factor = 0.6   # aggressive
    else:
        factor = 0.3   # defensive   
    attackers = sorted(my_planets, key=lambda x: -x.ships)[:3]
    target = max(
            opp_planets,
            key=lambda t: t.prod * 3 - t.ships - 0.2 * dist(t.x, t.y, CENTER, CENTER)
    )

    for s in attackers:
        reserve = max(10, s.prod * 2)
        send = int((s.ships - reserve) * factor)

        if send > 0:
            acts.append((s.id, intercept_angle(s, target, ANG_VEL, send), send, 0.0, target.id))
    return acts

# ======================
# PUCT
# ======================

def select(node):
    def score(child):
        q = child.value / (child.visits + 1e-6)
        q = max(-500, min(500, q))

        u = CPUCT * child.prior * math.sqrt(node.visits + 1) / (1 + child.visits * 0.8)
        return q + u

    return max(node.children, key=score)


# ======================
# EXPAND
# ======================

def expand(node):

    limit = int(6 + node.visits ** 0.5)
    if len(node.children) >= limit:
        return None

    if node.untried is None:

        def_acts = critical_defense(node, node.planets, node.fleets, node.player)

        if def_acts:
            sets = [def_acts]
        else:
            acts = gen_actions(node.planets, node.player)
            sets = gen_action_sets(node, acts, node.planets, node.fleets)
            if not sets:
                sets = [[]]
        def score_set(s):
            val = 0

            for (_, _, ships, _, tid) in s:
                
                t = next(p for p in node.planets if p.id == tid)
                val -= 0.3 * dist(t.x, t.y, CENTER, CENTER)

                val += ships
                val += 10 * t.prod
                val -= 0.7 * t.ships
                val -= 0.2 * ships
                
            targets = [a[4] for a in s]
            val += 15 * len(set(targets))

            targets = [a[4] for a in s]
            same_target_bonus = len(s) - len(set(targets))
            val += 25 * same_target_bonus
            total_sent = sum(a[2] for a in s)

            # don't drain planets
            val -= 0.15 * total_sent

            # stronger penalty late game
            if len(node.planets) < 12:
                val -= 0.25 * total_sent
            return val

        scored = [(score_set(s), s) for s in sets]
        temp = 0.4 if len(node.planets) > 15 else 0.25
        if not scored:
            node.untried = []
            return None

        max_p = max(p for p,_ in scored)

        exp_scores = [(math.exp((p - max_p) / temp), a) for p, a in scored]
        if not exp_scores:
            node.untried = []
            return None

        total = sum(p for p, _ in exp_scores)

        if total <= 1e-9:
            node.untried = [(1/len(exp_scores), a) for _, a in exp_scores]
        else:
            node.untried = [(p/total, a) for p, a in exp_scores]        # Dirichlet noise at root
        if node.parent is None:
            eps = 0.25
            alpha = 0.3

            noise = [random.gammavariate(alpha, 1) for _ in node.untried]
            total_n = sum(noise)
            noise = [n/total_n for n in noise]

            node.untried = [
                ((1-eps)*p + eps*n, a)
                for (p,a), n in zip(node.untried, noise)
            ]

    if not node.untried:
        return None

    r = random.random()
    cum = 0
    chosen = False

    for i, (p,a) in enumerate(node.untried):
        cum += p
        if r <= cum:
            node.untried.pop(i)
            prior, acts = p, a
            chosen = True
            break

    # fallback (ADD THIS)
    if not chosen:
        if not node.untried:
            return None
        prior, acts = node.untried.pop()

    p2, f2 = apply(node.planets, node.fleets, acts, node.player)

    enemy_act = enemy(p2, f2, node.player)
    p2, f2 = apply(p2, f2, enemy_act, 1-node.player)

    child = Node(p2, f2, 1-node.player, node, acts, prior)
    node.children.append(child)

    return child


# ======================
# BACKPROP
# ======================

def backprop(node, score, root_player):

    while node:
        node.visits += 1

        if node.player == root_player:
            node.value += score
        else:
            node.value -= score

        node = node.parent


# ======================
# MCTS
# ======================

def mcts(root):

    start = time.time()
    horizon = get_horizon(root.planets)

    while time.time() - start < TIME_LIMIT:

        node = root

        while node.children:
            node = select(node)

        child = expand(node)
        target = child if child else node

        sim_planets = target.planets
        sim_fleets = target.fleets

        depth = 4 if len(root.planets) < 10 else 3
        for i in range(depth):
            sim_planets = simulate(sim_planets, sim_fleets, horizon)

         
        score = evaluate(target, sim_planets, sim_fleets, root.player, horizon)

        backprop(target, score, root.player)
    if not root.children:
        return []
    best = max(
        root.children,
        key=lambda c: (c.visits, c.value/(c.visits+1e-6))
    )

    return best.action


# ======================
# AGENT
# ======================

def agent(obs):

    if len(DIST_CACHE) > 50000:
        DIST_CACHE.clear()
    if len(SPEED_CACHE) > 5000:
        SPEED_CACHE.clear()

    planets = [P(p) for p in obs["planets"]]
    fleets = [
        F(f[1], f[2], f[3], f[4], f[6], 0.0, -1)  # ships=f[6], target=-1 (unknown)
        for f in obs["fleets"]
    ]
    if random.random() < 0.01:
        print(obs["fleets"])
    global ANG_VEL
    ANG_VEL = obs.get("angular_velocity", 0.03)
    player = obs["player"]
    global ENEMY_GROWTH
    enemy_prod = sum(p.prod for p in planets if p.owner == 1-player)
    ENEMY_GROWTH = 0.7 * ENEMY_GROWTH + 0.3 * enemy_prod
    root = Node(planets, fleets, player)
    best = mcts(root)

    return best if best else []
