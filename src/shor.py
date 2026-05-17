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
def _(ClassicalRegister, QuantumCircuit, QuantumRegister, np):
    def prep_state(qc, q):
        qc.ry(np.pi / 4, q)
        qc.s(q)


    def inv_prep_state(qc, q):
        qc.sdg(q)
        qc.ry(-np.pi / 4, q)


    def encode_shor(qc, data):
        qc.cx(data[0], data[3])
        qc.cx(data[0], data[6])
        qc.h(data[0])
        qc.h(data[3])
        qc.h(data[6])
        for base in (0, 3, 6):
            qc.cx(data[base], data[base + 1])
            qc.cx(data[base], data[base + 2])


    def decode_shor(qc, data):
        for base in (0, 3, 6):
            qc.cx(data[base], data[base + 2])
            qc.cx(data[base], data[base + 1])
        qc.h(data[0])
        qc.h(data[3])
        qc.h(data[6])
        qc.cx(data[0], data[6])
        qc.cx(data[0], data[3])


    def ccz(qc, c1, c2, t):
        qc.h(t)
        qc.ccx(c1, c2, t)
        qc.h(t)


    def bit_flip_extract_and_correct(qc, data, base, a0, a1):
        qc.cx(data[base], a0)
        qc.cx(data[base + 1], a0)
        qc.cx(data[base + 1], a1)
        qc.cx(data[base + 2], a1)
        qc.x(a1)
        qc.ccx(a0, a1, data[base])
        qc.x(a1)
        qc.ccx(a0, a1, data[base + 1])
        qc.x(a0)
        qc.ccx(a0, a1, data[base + 2])
        qc.x(a0)


    def phase_flip_extract_and_correct(qc, data, a0, a1):
        qc.h(a0)
        for i in range(0, 6):
            qc.cx(a0, data[i])
        qc.h(a0)
        qc.h(a1)
        for i in range(3, 9):
            qc.cx(a1, data[i])
        qc.h(a1)
        # Deferred Z corrections.
        qc.x(a1)
        ccz(qc, a0, a1, data[0])
        qc.x(a1)
        ccz(qc, a0, a1, data[3])
        qc.x(a0)
        ccz(qc, a0, a1, data[6])
        qc.x(a0)


    def basis_rotation(qc, q, basis):
        if basis == "X":
            qc.h(q)
        elif basis == "Y":
            qc.sdg(q)
            qc.h(q)
        elif basis != "Z":
            raise ValueError(f"unknown basis {basis}")


    def build_shor_circuit(basis="Z"):
        data = QuantumRegister(9, "data")
        ba = QuantumRegister(6, "ba")
        pa = QuantumRegister(2, "pa")
        out = ClassicalRegister(1, "out")
        qc = QuantumCircuit(data, ba, pa, out)

        prep_state(qc, data[0])
        qc.barrier()
        encode_shor(qc, data)
        qc.barrier()

        for i in range(9):
            qc.id(data[i])
        qc.barrier()

        for block_idx, base in enumerate((0, 3, 6)):
            bit_flip_extract_and_correct(
                qc, data, base, ba[2 * block_idx], ba[2 * block_idx + 1]
            )
        qc.barrier()

        phase_flip_extract_and_correct(qc, data, pa[0], pa[1])
        qc.barrier()

        decode_shor(qc, data)
        if basis == "Z":
            inv_prep_state(qc, data[0])
        else:
            basis_rotation(qc, data[0], basis)
        qc.measure(data[0], out[0])

        return qc


    def build_baseline_circuit(basis="Z"):
        data = QuantumRegister(1, "data")
        out = ClassicalRegister(1, "out")
        qc = QuantumCircuit(data, out)
        prep_state(qc, data[0])
        qc.barrier()
        qc.id(data[0])
        qc.barrier()
        if basis == "Z":
            inv_prep_state(qc, data[0])
        else:
            basis_rotation(qc, data[0], basis)
        qc.measure(data[0], out[0])
        return qc

    return build_baseline_circuit, build_shor_circuit


@app.cell
def _(build_shor_circuit):
    build_shor_circuit().draw(output="mpl", fold=-1)
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
    sim = AerSimulator()


    def transpile_for_sim(qc):
        return transpile(qc, sim, optimization_level=0)


    def evaluate_with_noise(tqc, noise_model, shots=2000):
        result = sim.run(tqc, shots=int(shots), noise_model=noise_model).result()
        return result.get_counts()

    return evaluate_with_noise, transpile_for_sim


@app.cell
def _(mo):
    p_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.01, value=0.1, label="Bit-flip probability p"
    )
    shots_slider = mo.ui.slider(start=100, stop=10000, step=100, value=2000, label="Shots")
    mo.vstack([p_slider, shots_slider])
    return p_slider, shots_slider


@app.cell
def _(
    build_baseline_circuit,
    build_shor_circuit,
    evaluate_with_noise,
    make_depolarizing_noise_model,
    mo,
    p_slider,
    shots_slider,
    transpile_for_sim,
):
    def err_rate(counts):
        total = sum(counts.values())
        errs = sum(n for key, n in counts.items() if key.split()[0] == "1")
        return errs, total, (errs / total if total else 0.0)


    table_p = float(p_slider.value)
    table_shots = int(shots_slider.value)
    table_nm = make_depolarizing_noise_model(table_p)
    shor_counts = evaluate_with_noise(
        transpile_for_sim(build_shor_circuit()), table_nm, shots=table_shots
    )
    base_counts = evaluate_with_noise(
        transpile_for_sim(build_baseline_circuit()), table_nm, shots=table_shots
    )
    shor_err, shor_total, shor_p_err = err_rate(shor_counts)
    base_err, base_total, base_p_err = err_rate(base_counts)

    mo.md(
        f"""
    ### Logical error at p = {table_p:.3f}, {table_shots} shots

    | scenario | errors / shots | P(error) |
    |---|---|---|
    | single qubit (no code) | {base_err} / {base_total} | **{base_p_err:.4f}** |
    | Shor [[9,1,3]] code    | {shor_err} / {shor_total} | **{shor_p_err:.4f}** |
    """
    )
    return


@app.cell
def _(mo):
    sweep_points_slider = mo.ui.slider(
        start=5, stop=99, step=1, value=21, label="Sweep points"
    )
    sweep_points_slider
    return (sweep_points_slider,)


@app.cell
def _(
    build_baseline_circuit,
    build_shor_circuit,
    evaluate_with_noise,
    make_depolarizing_noise_model,
    np,
    shots_slider,
    sweep_points_slider,
    transpile_for_sim,
):
    n_points = int(sweep_points_slider.value)
    shots = int(shots_slider.value)
    ps = np.linspace(0.01, 0.3, n_points)


    def err_rate_nm(tqc, nm, shots):
        counts = evaluate_with_noise(tqc, nm, shots=shots)
        total = sum(counts.values())
        errs = sum(n for key, n in counts.items() if key.split()[0] == "1")
        return errs / total if total else 0.0


    base_tqc = transpile_for_sim(build_baseline_circuit())
    shor_tqc = transpile_for_sim(build_shor_circuit())
    shor_baseline = np.array(
        [err_rate_nm(base_tqc, make_depolarizing_noise_model(p), shots) for p in ps]
    )
    shor_coded = np.array(
        [err_rate_nm(shor_tqc, make_depolarizing_noise_model(p), shots) for p in ps]
    )
    return ps, shor_baseline, shor_coded, shots


@app.cell
def _(plt, ps, shor_baseline, shor_coded):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ps, shor_baseline, "o-", label="single qubit (no code)")
    ax.plot(ps, shor_coded, "s-", label="Shor [[9,1,3]] code")
    ax.set_xlabel("depolarizing channel strength p")
    ax.set_ylabel("measured logical error rate")
    ax.set_title("Shor code vs. single qubit under depolarizing noise")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig
    return


@app.cell
def _(
    build_baseline_circuit,
    build_shor_circuit,
    evaluate_with_noise,
    make_depolarizing_noise_model,
    np,
    ps,
    shots,
    transpile_for_sim,
):
    def expectation(tqc, nm, shots):
        counts = evaluate_with_noise(tqc, nm, shots=shots)
        total = sum(counts.values())
        n0 = sum(n for key, n in counts.items() if key.split()[0] == "0")
        n1 = total - n0
        return (n0 - n1) / total if total else 0.0


    def sweep_expectation(tqc, ps, shots):
        return np.array(
            [expectation(tqc, make_depolarizing_noise_model(p), shots) for p in ps]
        )


    bases = ("X", "Y", "Z")
    bloch_base = {
        b: sweep_expectation(transpile_for_sim(build_baseline_circuit(b)), ps, shots)
        for b in bases
    }
    bloch_shor = {
        b: sweep_expectation(transpile_for_sim(build_shor_circuit(basis=b)), ps, shots)
        for b in bases
    }
    return bloch_base, bloch_shor


@app.cell
def _(bloch_base, bloch_shor, np, plt, ps):
    ideal = {"X": 0.0, "Y": np.sqrt(2) / 2, "Z": np.sqrt(2) / 2}

    fig_t, axes_t = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax_t, basis in zip(axes_t, ("X", "Y", "Z")):
        ax_t.axhline(
            ideal[basis], color="k", linestyle=":", alpha=0.5, label=f"ideal ⟨{basis}⟩"
        )
        ax_t.plot(ps, bloch_base[basis], "o-", label="single qubit")
        ax_t.plot(ps, bloch_shor[basis], "s-", label="Shor [[9,1,3]] code")
        ax_t.set_xlabel("depolarizing strength p")
        ax_t.set_title(f"⟨{basis}⟩ on decoded qubit")
        ax_t.grid(alpha=0.3)
        ax_t.legend()
    axes_t[0].set_ylabel("expectation value")
    fig_t.tight_layout()
    fig_t
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
