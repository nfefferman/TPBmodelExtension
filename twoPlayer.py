def _run_single_sim(args):
    a1, a2, sim_kwargs = args
    
    n = 2
    t_max = sim_kwargs['t_max']
    threshold_x = sim_kwargs['threshold_x']
    
    alpha_arr = np.array([a1, a2], dtype=float)
    x0 = np.array([sim_kwargs['x_init']] * n, dtype=float)
    y0 = np.array([sim_kwargs['y_init']] * n, dtype=float)
    
    def rhs(t, state):
        x = state[:n]
        y = state[n:]
        
        r = 0.86
        mu_c = 0.05
        sigma_a = 0.5
        sigma_s = 0.5
        sigma_c = 0.5
        mu_s = -0.5
        
        gamma = (np.sum(y) - y) / (n - 1) if n > 1 else np.zeros(n)
            
        dxdt = (sigma_a * alpha_arr + sigma_s * (gamma - mu_s)) * sigma_c * (gamma + mu_c) * (1 - x) * (1 + x)
        dydt = -r * y 
        
        return np.concatenate([dxdt, dydt])

    def hit_threshold(t, state):
        return np.max(state[:n]) - threshold_x
    
    hit_threshold.terminal = True 
    hit_threshold.direction = 1 

    t_current = 0.0
    state = np.concatenate([x0, y0])
    
    has_acted = np.zeros(n, dtype=bool)
    
    events_triggered = 0
    max_events = 1000 
    
    while t_current < t_max and events_triggered < max_events:
        sol = solve_ivp(
            fun=rhs,
            t_span=(t_current, t_max),
            y0=state,
            method='RK45',      
            events=hit_threshold,
            rtol=1e-5,          
            atol=1e-7           
        )
        
        if sol.status == 1: 
            t_current = sol.t[-1]
            state = sol.y[:, -1].copy()
            
            x_current = state[:n]
            y_current = state[n:] 
            
            players_to_reset = x_current >= (threshold_x - 1e-8) 
            has_acted[players_to_reset] = True
            
            if np.all(has_acted):
                break
            
            x_current[players_to_reset] = 0 
            y_current[players_to_reset] = 1.0
            
            state[:n] = x_current
            state[n:] = y_current
            events_triggered += 1
            
        else:
            break

    flag = int(np.sum(has_acted))

    return {
        "alpha_1": a1,
        "alpha_2": a2,
        "flag": flag,
    }

# Sweep
def generate_heatmap_csv(
    grid_size: int = 100, 
    t_max: float = 5000,
    save_path: str = "alpha_sweep_2player.csv"
):
    alpha_vals = np.round(np.linspace(-1.0, 1.0, grid_size), 6)
    
    sim_kwargs = {
        "t_max": t_max,
        "threshold_x": 0.8,
        "x_init": 0.0,
        "y_init": 0.0
    }

    unique_pairs = list(itertools.combinations_with_replacement(alpha_vals, 2))
    tasks = [(a1, a2, sim_kwargs) for a1, a2 in unique_pairs]

    print(f"Starting {len(tasks)} simulations in parallel")
    
    # parallelization
    results = Parallel(n_jobs=-1)(delayed(_run_single_sim)(task) for task in tasks)
    
    rows = []
    for res in results:
        rows.append(res)
        # Mirror the data by symmetry
        if res["alpha_1"] != res["alpha_2"]:
            rows.append({
                "alpha_1": res["alpha_2"],
                "alpha_2": res["alpha_1"],
                "flag": res["flag"]
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["alpha_1", "alpha_2"]).reset_index(drop=True)
    
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    
    print(f"Saved full {grid_size}x{grid_size} grid ({len(df)} rows) to: {out.resolve()}")
    return df