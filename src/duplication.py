import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator

    from qiskit_aer.noise import NoiseModel, depolarizing_error



    return (
        AerSimulator,
        ClassicalRegister,
        NoiseModel,
        QuantumCircuit,
        QuantumRegister,
        depolarizing_error,
        mo,
        np,
        plt,
        transpile,
    )


@app.cell
def _(mo):
    p_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.01, value=0.1, label="Bit-flip probability p"
    )
    shots_slider = mo.ui.slider(start=100, stop=10000, step=100, value=2000, label="Shots")
    mo.vstack([p_slider, shots_slider])
    return p_slider, shots_slider


@app.cell
def _(ClassicalRegister, QuantumCircuit, QuantumRegister, np):
    def build_circuit(p, record_noise=True):
        data = QuantumRegister(3, "data")
        noise = QuantumRegister(3, "noise")
        anc = QuantumRegister(2, "anc")
        syn = ClassicalRegister(2, "syndrome")

        out = ClassicalRegister(1, "out")
        regs = [data, noise, anc, syn, out]
        if record_noise:
            err = ClassicalRegister(3, "err")
            regs.append(err)
        qc = QuantumCircuit(*regs)

        qc.ry(np.pi / 4, data[0])
        qc.s(data[0])
        qc.barrier()

        qc.cx(data[0], data[1])
        qc.cx(data[0], data[2])
        qc.barrier()

        theta = 2.0 * np.arcsin(np.sqrt(p))
        for i in range(3):
            qc.ry(theta, noise[i])
            qc.cx(noise[i], data[i])
        if record_noise:
            for i in range(3):
                qc.measure(noise[i], err[i])
        qc.barrier()

        qc.cx(data[0], anc[0])
        qc.cx(data[1], anc[0])
        qc.cx(data[1], anc[1])
        qc.cx(data[2], anc[1])
        qc.barrier()

        qc.measure(anc[0], syn[0])
        qc.measure(anc[1], syn[1])
        qc.barrier()

        with qc.if_test((syn, 1)):
            qc.x(data[0])
        with qc.if_test((syn, 3)):
            qc.x(data[1])
        with qc.if_test((syn, 2)):
            qc.x(data[2])
        qc.barrier()

        qc.cx(data[0], data[2])
        qc.cx(data[0], data[1])

        qc.sdg(data[0])
        qc.ry(-np.pi / 4, data[0])
        qc.measure(data[0], out[0])

        return qc

    return (build_circuit,)


@app.cell
def _(ClassicalRegister, QuantumCircuit, QuantumRegister, np):
    def build_baseline_circuit(p):
        data = QuantumRegister(1, "data")
        noise = QuantumRegister(1, "noise")
        out = ClassicalRegister(1, "out")
        qc = QuantumCircuit(data, noise, out)

        # Same state prep as the encoded version.
        qc.ry(np.pi / 4, data[0])
        qc.s(data[0])
        qc.barrier()

        # Bit-flip channel via Stinespring dilation.
        theta = 2.0 * np.arcsin(np.sqrt(p))
        qc.ry(theta, noise[0])
        qc.cx(noise[0], data[0])
        qc.barrier()

        # Inverse state prep, then measure: 0 = no logical error, 1 = error.
        qc.sdg(data[0])
        qc.ry(-np.pi / 4, data[0])
        qc.measure(data[0], out[0])

        return qc

    return (build_baseline_circuit,)


@app.cell
def _(ClassicalRegister, QuantumCircuit, QuantumRegister, np):
    def build_phase_flip_circuit(p):
        data = QuantumRegister(3, "data")
        noise = QuantumRegister(3, "noise")
        anc = QuantumRegister(2, "anc")
        syn = ClassicalRegister(2, "syndrome")
        out = ClassicalRegister(1, "out")
        qc = QuantumCircuit(data, noise, anc, syn, out)

        qc.ry(np.pi / 4, data[0])
        qc.s(data[0])
        qc.barrier()

        qc.cx(data[0], data[1])
        qc.cx(data[0], data[2])
        qc.h(data[0])
        qc.h(data[1])
        qc.h(data[2])
        qc.barrier()

        theta = 2.0 * np.arcsin(np.sqrt(p))
        for i in range(3):
            qc.ry(theta, noise[i])
            qc.cz(noise[i], data[i])
        qc.barrier()

        qc.h(data[0])
        qc.h(data[1])
        qc.h(data[2])
        qc.barrier()

        qc.cx(data[0], anc[0])
        qc.cx(data[1], anc[0])
        qc.cx(data[1], anc[1])
        qc.cx(data[2], anc[1])
        qc.measure(anc[0], syn[0])
        qc.measure(anc[1], syn[1])
        qc.barrier()

        with qc.if_test((syn, 1)):
            qc.x(data[0])
        with qc.if_test((syn, 3)):
            qc.x(data[1])
        with qc.if_test((syn, 2)):
            qc.x(data[2])
        qc.barrier()

        # Decode and measure logical error.
        qc.cx(data[0], data[2])
        qc.cx(data[0], data[1])
        qc.sdg(data[0])
        qc.ry(-np.pi / 4, data[0])
        qc.measure(data[0], out[0])

        return qc

    return (build_phase_flip_circuit,)


@app.cell
def _(ClassicalRegister, QuantumCircuit, QuantumRegister, np):
    def build_baseline_z_circuit(p):
        data = QuantumRegister(1, "data")
        noise = QuantumRegister(1, "noise")
        out = ClassicalRegister(1, "out")
        qc = QuantumCircuit(data, noise, out)

        qc.ry(np.pi / 4, data[0])
        qc.s(data[0])
        qc.barrier()

        theta = 2.0 * np.arcsin(np.sqrt(p))
        qc.ry(theta, noise[0])
        qc.cz(noise[0], data[0])
        qc.barrier()

        qc.sdg(data[0])
        qc.ry(-np.pi / 4, data[0])
        qc.measure(data[0], out[0])

        return qc

    return (build_baseline_z_circuit,)


@app.cell
def _(AerSimulator, transpile):
    def evaluate(qc, shots=2000):
        sim = AerSimulator()
        tqc = transpile(qc, sim)
        result = sim.run(tqc, shots=int(shots)).result()
        return result.get_counts()

    return (evaluate,)


@app.cell
def _(build_circuit, p_slider):
    qc = build_circuit(float(p_slider.value))
    qc.draw(output="mpl", fold=-1)
    return (qc,)


@app.cell
def _(build_baseline_circuit, p_slider):
    qc_baseline = build_baseline_circuit(p_slider.value)
    qc_baseline.draw("mpl")
    return


@app.cell
def _(build_baseline_z_circuit, p_slider):
    qc_baseline_z = build_baseline_z_circuit(p_slider.value)
    qc_baseline_z.draw("mpl")
    return


@app.cell
def _(evaluate, qc, shots_slider):
    counts = evaluate(qc, shots=shots_slider.value)
    return (counts,)


@app.cell
def _(build_baseline_circuit, evaluate, p_slider, shots_slider):
    baseline_qc = build_baseline_circuit(float(p_slider.value))
    baseline_counts = evaluate(baseline_qc, shots=shots_slider.value)
    baseline_total = sum(baseline_counts.values())
    baseline_errors = baseline_counts.get("1", 0)
    p_baseline = baseline_errors / baseline_total if baseline_total else 0.0
    return baseline_errors, baseline_total, p_baseline


@app.cell
def _(baseline_errors, baseline_total, counts, mo, p_baseline):
    diagnosis = {
        "00": "no error (or error on all three)",
        "01": "error on data[0]",
        "11": "error on data[1]",
        "10": "error on data[2]",
    }

    syn_counts = {"00": 0, "01": 0, "10": 0, "11": 0}
    err_counts = {}
    total = 0
    logical_errors = 0
    for key, n in counts.items():
        parts = key.split()
        # rightmost = syndrome (first-registered)
        s = parts[-1]
        o = parts[-2]
        e = parts[0] if len(parts) >= 3 else ""
        syn_counts[s] = syn_counts.get(s, 0) + n
        if e:
            err_counts[e] = err_counts.get(e, 0) + n
        total += n
        if o == "1":
            logical_errors += n

    p_logical = logical_errors / total if total else 0.0

    syn_rows = "\n".join(
        f"| `{s}` | {syn_counts.get(s, 0)} | {diagnosis[s]} |"
        for s in ["00", "01", "10", "11"]
    )

    if err_counts:
        err_rows = "\n".join(
            f"| `{e}` | {n} |" for e, n in sorted(err_counts.items(), key=lambda kv: -kv[1])
        )
        err_section = f"""
    ### Actual X-errors per shot (noise register, bits `e2 e1 e0`)

    | err pattern | count |
    |---|---|
    {err_rows}
    """
    else:
        err_section = ""

    mo.md(
        f"""
    ### Error probabilities

    | scenario | errors / shots | P(error) |
    |---|---|---|
    | Uncorrected single qubit (baseline) | {baseline_errors} / {baseline_total} | **{p_baseline:.4f}** |
    | 3-qubit bit-flip code with correction | {logical_errors} / {total} | **{p_logical:.4f}** |

    The code helps when `p_logical < p_baseline` (below the threshold p ≈ 0.5 for this code).

    ### Syndrome measurement results

    | syndrome (s1 s0) | count | diagnosis |
    |---|---|---|
    {syn_rows}
    {err_section}
    """
    )
    return


@app.cell
def _(mo):
    sweep_shots_slider = mo.ui.slider(
        start=200, stop=5000, step=100, value=2000, label="Sweep shots per p"
    )
    sweep_points_slider = mo.ui.slider(
        start=10, stop=99, step=1, value=99, label="Sweep points"
    )
    mo.vstack([sweep_shots_slider, sweep_points_slider])
    return sweep_points_slider, sweep_shots_slider


@app.cell
def _(
    build_baseline_circuit,
    build_baseline_z_circuit,
    build_circuit,
    build_phase_flip_circuit,
    evaluate,
    np,
    sweep_points_slider,
    sweep_shots_slider,
):
    n_points = int(sweep_points_slider.value)
    shots = int(sweep_shots_slider.value)
    ps = np.linspace(0.01, 0.99, n_points)


    def _err_rate(qc, shots):
        counts = evaluate(qc, shots=shots)
        total = sum(counts.values())

        errs = sum(n for key, n in counts.items() if key.split()[0] == "1")
        return errs / total if total else 0.0


    bit_baseline = np.array([_err_rate(build_baseline_circuit(p), shots) for p in ps])
    bit_coded = np.array(
        [_err_rate(build_circuit(p, record_noise=False), shots) for p in ps]
    )
    phase_baseline = np.array([_err_rate(build_baseline_z_circuit(p), shots) for p in ps])
    phase_coded = np.array([_err_rate(build_phase_flip_circuit(p), shots) for p in ps])
    return (
        bit_baseline,
        bit_coded,
        n_points,
        phase_baseline,
        phase_coded,
        ps,
        shots,
    )


@app.cell
def _(bit_baseline, bit_coded, phase_baseline, phase_coded, plt, ps):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ps, bit_baseline, "o-", label="single qubit (no code)")
    axes[0].plot(ps, bit_coded, "s-", label="3-qubit bit-flip code")
    axes[0].plot(ps, ps, "m--", label="y = p")
    axes[0].set_xlabel("physical bit-flip probability p")
    axes[0].set_ylabel("measured logical error rate")
    axes[0].set_title("Bit-flip channel (X noise)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ps, phase_baseline, "o-", label="single qubit (no code)")
    axes[1].plot(ps, phase_coded, "s-", label="3-qubit phase-flip code")
    axes[1].set_xlabel("physical phase-flip probability p")
    axes[1].set_ylabel("measured logical error rate")
    axes[1].set_title("Phase-flip channel (Z noise)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig
    return


@app.cell
def _(ClassicalRegister, QuantumCircuit, QuantumRegister, np):
    def build_clean_bit_flip_circuit():
        data = QuantumRegister(3, "data")
        anc = QuantumRegister(2, "anc")
        syn = ClassicalRegister(2, "syndrome")
        out = ClassicalRegister(1, "out")
        qc = QuantumCircuit(data, anc, syn, out)

        qc.ry(np.pi / 4, data[0])
        qc.s(data[0])
        qc.barrier()

        qc.cx(data[0], data[1])
        qc.cx(data[0], data[2])
        qc.barrier()

        for i in range(3):
            qc.id(data[i])
        qc.barrier()

        qc.cx(data[0], anc[0])
        qc.cx(data[1], anc[0])
        qc.cx(data[1], anc[1])
        qc.cx(data[2], anc[1])
        qc.measure(anc[0], syn[0])
        qc.measure(anc[1], syn[1])
        qc.barrier()

        with qc.if_test((syn, 1)):
            qc.x(data[0])
        with qc.if_test((syn, 3)):
            qc.x(data[1])
        with qc.if_test((syn, 2)):
            qc.x(data[2])
        qc.barrier()

        qc.cx(data[0], data[2])
        qc.cx(data[0], data[1])
        qc.sdg(data[0])
        qc.ry(-np.pi / 4, data[0])
        qc.measure(data[0], out[0])

        return qc


    def build_clean_baseline_circuit():
        data = QuantumRegister(1, "data")
        out = ClassicalRegister(1, "out")
        qc = QuantumCircuit(data, out)

        qc.ry(np.pi / 4, data[0])
        qc.s(data[0])
        qc.barrier()
        qc.id(data[0])
        qc.barrier()
        qc.sdg(data[0])
        qc.ry(-np.pi / 4, data[0])
        qc.measure(data[0], out[0])

        return qc

    return build_clean_baseline_circuit, build_clean_bit_flip_circuit


@app.cell
def _(build_clean_bit_flip_circuit):
    build_clean_bit_flip_circuit().draw("mpl", fold=-1)
    return


@app.cell
def _(build_clean_baseline_circuit):
    build_clean_baseline_circuit().draw("mpl", fold=-1)
    return


@app.cell
def _(NoiseModel, depolarizing_error):
    def make_depolarizing_noise_model(p):
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["id"])
        return nm

    return (make_depolarizing_noise_model,)


@app.cell
def _(AerSimulator, transpile):
    def evaluate_with_noise(qc, noise_model, shots=2000):
        sim = AerSimulator(noise_model=noise_model)
        tqc = transpile(qc, sim, optimization_level=0)
        result = sim.run(tqc, shots=int(shots)).result()
        return result.get_counts()

    return (evaluate_with_noise,)


@app.cell
def _(
    build_clean_baseline_circuit,
    build_clean_bit_flip_circuit,
    evaluate_with_noise,
    make_depolarizing_noise_model,
    n_points,
    np,
    shots,
):
    ps_r = np.linspace(0.01, 0.99, n_points)


    def _err_rate_nm(qc, nm, shots):
        counts = evaluate_with_noise(qc, nm, shots=shots)
        total = sum(counts.values())
        errs = sum(n for key, n in counts.items() if key.split()[0] == "1")
        return errs / total if total else 0.0


    base_qc = build_clean_baseline_circuit()
    code_qc = build_clean_bit_flip_circuit()
    real_baseline = np.array(
        [_err_rate_nm(base_qc, make_depolarizing_noise_model(p), shots) for p in ps_r]
    )
    real_coded = np.array(
        [_err_rate_nm(code_qc, make_depolarizing_noise_model(p), shots) for p in ps_r]
    )
    return ps_r, real_baseline, real_coded


@app.cell
def _(plt, ps_r, real_baseline, real_coded):
    fig_real, ax_real = plt.subplots(figsize=(7, 5))
    ax_real.plot(ps_r, real_baseline, "o-", label="single qubit (no code)")
    ax_real.plot(ps_r, real_coded, "s-", label="3-qubit bit-flip code")
    ax_real.set_xlabel("depolarizing channel strength p")
    ax_real.set_ylabel("measured logical error rate")
    ax_real.set_title("Realistic noise: depolarizing channel + bit-flip code")
    ax_real.legend()
    ax_real.grid(alpha=0.3)
    fig_real.tight_layout()
    fig_real
    return


@app.cell
def _(ClassicalRegister, QuantumCircuit, QuantumRegister, np):
    def _basis_rotation(qc, qubit, basis):
        if basis == "X":
            qc.h(qubit)
        elif basis == "Y":
            qc.sdg(qubit)
            qc.h(qubit)
        elif basis != "Z":
            raise ValueError(f"unknown basis {basis}")


    def build_clean_bit_flip_basis(basis):
        data = QuantumRegister(3, "data")
        anc = QuantumRegister(2, "anc")
        syn = ClassicalRegister(2, "syndrome")
        out = ClassicalRegister(1, "out")
        qc = QuantumCircuit(data, anc, syn, out)

        qc.ry(np.pi / 4, data[0])
        qc.s(data[0])
        qc.barrier()

        qc.cx(data[0], data[1])
        qc.cx(data[0], data[2])
        qc.barrier()

        for i in range(3):
            qc.id(data[i])
        qc.barrier()

        qc.cx(data[0], anc[0])
        qc.cx(data[1], anc[0])
        qc.cx(data[1], anc[1])
        qc.cx(data[2], anc[1])
        qc.measure(anc[0], syn[0])
        qc.measure(anc[1], syn[1])
        qc.barrier()

        with qc.if_test((syn, 1)):
            qc.x(data[0])
        with qc.if_test((syn, 3)):
            qc.x(data[1])
        with qc.if_test((syn, 2)):
            qc.x(data[2])
        qc.barrier()

        qc.cx(data[0], data[2])
        qc.cx(data[0], data[1])

        _basis_rotation(qc, data[0], basis)
        qc.measure(data[0], out[0])
        return qc


    def build_clean_baseline_basis(basis):
        data = QuantumRegister(1, "data")
        out = ClassicalRegister(1, "out")
        qc = QuantumCircuit(data, out)
        qc.ry(np.pi / 4, data[0])
        qc.s(data[0])
        qc.barrier()
        qc.id(data[0])
        qc.barrier()
        _basis_rotation(qc, data[0], basis)
        qc.measure(data[0], out[0])
        return qc

    return build_clean_baseline_basis, build_clean_bit_flip_basis


@app.cell
def _(build_clean_bit_flip_basis):
    build_clean_bit_flip_basis("X").draw("mpl")
    return


@app.cell
def _(build_clean_bit_flip_basis):
    build_clean_bit_flip_basis("Y").draw("mpl")
    return


@app.cell
def _(build_clean_bit_flip_basis):
    build_clean_bit_flip_basis("Z").draw("mpl")
    return


@app.cell
def _(build_clean_baseline_basis):
    build_clean_baseline_basis("X").draw("mpl")
    return


@app.cell
def _(build_clean_baseline_basis):
    build_clean_baseline_basis("Y").draw("mpl")
    return


@app.cell
def _(build_clean_baseline_basis):
    build_clean_baseline_basis("Z").draw("mpl")
    return


@app.cell
def _(
    build_clean_baseline_basis,
    build_clean_bit_flip_basis,
    evaluate_with_noise,
    make_depolarizing_noise_model,
    np,
    ps,
    shots,
):
    def _expectation(qc, nm, shots):
        counts = evaluate_with_noise(qc, nm, shots=shots)
        total = sum(counts.values())
        n0 = sum(n for key, n in counts.items() if key.split()[0] == "0")
        n1 = total - n0
        return (n0 - n1) / total if total else 0.0


    base_qcs = {b: build_clean_baseline_basis(b) for b in ("X", "Y", "Z")}
    code_qcs = {b: build_clean_bit_flip_basis(b) for b in ("X", "Y", "Z")}

    bloch_base = {b: [] for b in ("X", "Y", "Z")}
    bloch_code = {b: [] for b in ("X", "Y", "Z")}
    for p in ps:
        nm = make_depolarizing_noise_model(p)
        for b in ("X", "Y", "Z"):
            bloch_base[b].append(_expectation(base_qcs[b], nm, shots))
            bloch_code[b].append(_expectation(code_qcs[b], nm, shots))
    bloch_base = {b: np.array(v) for b, v in bloch_base.items()}
    bloch_code = {b: np.array(v) for b, v in bloch_code.items()}
    return bloch_base, bloch_code


@app.cell
def _(bloch_base, bloch_code, np, plt, ps):
    ideal = {"X": 0.0, "Y": np.sqrt(2) / 2, "Z": np.sqrt(2) / 2}

    fig_t, axes_t = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, basis in zip(axes_t, ("X", "Y", "Z")):
        ax.axhline(
            ideal[basis], color="m", linestyle=":", label=f"ideal ⟨{basis}⟩"
        )
        ax.plot(ps, bloch_base[basis], "o-", label="single qubit")
        ax.plot(ps, bloch_code[basis], "s-", label="bit-flip code")
        ax.set_xlabel("depolarizing strength p")
        ax.set_title(f"⟨{basis}⟩ on decoded qubit")
        ax.grid(alpha=0.3)
        ax.legend()
    axes_t[0].set_ylabel("expectation value")
    fig_t.tight_layout()
    fig_t
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
