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

    return (
        AerSimulator,
        ClassicalRegister,
        QuantumCircuit,
        QuantumRegister,
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
        """Build the 3-qubit bit-flip code circuit.

        Each data qubit is sent through a bit-flip channel with probability p,
        implemented by entangling it with a fresh "environment" ancilla rotated
        to sqrt(1-p)|0> + sqrt(p)|1>. The simulator samples this per shot,
        so probability p is realised statistically across shots.

        If record_noise=True, the noise ancillas are measured into a separate
        3-bit classical register so we can see which qubits were flipped on
        each shot.
        """
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

        # |psi> = cos(pi/8)|0> + e^{i pi/2} sin(pi/8)|1>
        # Ry(2*pi/8) gives cos(pi/8)|0> + sin(pi/8)|1>; S adds the i phase on |1>.
        qc.ry(np.pi / 4, data[0])
        qc.s(data[0])
        qc.barrier()

        # Encode: a|0> + b|1>  ->  a|000> + b|111>
        qc.cx(data[0], data[1])
        qc.cx(data[0], data[2])
        qc.barrier()

        # Stochastic bit-flip channel via Stinespring dilation:
        # rotate each noise ancilla to sqrt(1-p)|0> + sqrt(p)|1>, then CNOT it
        # into the corresponding data qubit. Per shot, the simulator samples
        # the ancilla; if it collapses to |1>, the data qubit gets an X.
        theta = 2.0 * np.arcsin(np.sqrt(p))
        for i in range(3):
            qc.ry(theta, noise[i])
            qc.cx(noise[i], data[i])
        if record_noise:
            for i in range(3):
                qc.measure(noise[i], err[i])
        qc.barrier()

        # Syndrome extraction: Z0Z1 -> anc[0], Z1Z2 -> anc[1]
        qc.cx(data[0], anc[0])
        qc.cx(data[1], anc[0])
        qc.cx(data[1], anc[1])
        qc.cx(data[2], anc[1])
        qc.barrier()

        qc.measure(anc[0], syn[0])
        qc.measure(anc[1], syn[1])
        qc.barrier()

        # Correction conditioned on syndrome.
        # syn integer value = syn[1]*2 + syn[0]:
        #   1 (01) -> X on data[0],  3 (11) -> X on data[1],  2 (10) -> X on data[2]
        with qc.if_test((syn, 1)):
            qc.x(data[0])
        with qc.if_test((syn, 3)):
            qc.x(data[1])
        with qc.if_test((syn, 2)):
            qc.x(data[2])
        qc.barrier()

        # Decode (uncompute encoder) so data[0] holds the logical qubit.
        qc.cx(data[0], data[2])
        qc.cx(data[0], data[1])

        # Inverse of state prep on data[0]: Sdg then Ry(-pi/4).
        # If correction succeeded, data[0] is back to |psi>, and the inverse
        # maps it to |0>; a logical X (uncorrectable 2- or 3-qubit error)
        # leaves data[0] in a state whose inverse-prep gives |1> with prob 1.
        qc.sdg(data[0])
        qc.ry(-np.pi / 4, data[0])
        qc.measure(data[0], out[0])

        return qc

    return (build_circuit,)


@app.cell
def _(ClassicalRegister, QuantumCircuit, QuantumRegister, np):
    def build_baseline_circuit(p):
        """Single physical qubit, no encoding/correction. Bit-flip with prob p.

        Used as the uncorrected baseline: P(error) should be ~ p.
        """
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
        """3-qubit phase-flip code with Z noise and syndrome-based correction.

        Same structure as the bit-flip code, conjugated by Hadamards on the
        data qubits: encode in the X basis, suffer Z errors, then rotate back
        so Z errors become X errors and the bit-flip syndrome / correction
        machinery applies unchanged.
        """
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
        """Single physical qubit, Z noise with probability p, no correction."""
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
    qc.draw(output="mpl")
    return (qc,)


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
        start=200, stop=5000, step=100, value=1000, label="Sweep shots per p"
    )
    sweep_points_slider = mo.ui.slider(
        start=10, stop=99, step=1, value=25, label="Sweep points"
    )
    run_sweep_btn = mo.ui.run_button(label="Run sweep")
    mo.vstack([sweep_shots_slider, sweep_points_slider, run_sweep_btn])
    return run_sweep_btn, sweep_points_slider, sweep_shots_slider


@app.cell
def _(
    build_baseline_circuit,
    build_baseline_z_circuit,
    build_circuit,
    build_phase_flip_circuit,
    evaluate,
    np,
    run_sweep_btn,
    sweep_points_slider,
    sweep_shots_slider,
):
    mo_stop = not run_sweep_btn.value
    if mo_stop:
        ps = np.array([])
        bit_baseline = np.array([])
        bit_coded = np.array([])
        phase_baseline = np.array([])
        phase_coded = np.array([])
    else:
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
        phase_baseline = np.array(
            [_err_rate(build_baseline_z_circuit(p), shots) for p in ps]
        )
        phase_coded = np.array([_err_rate(build_phase_flip_circuit(p), shots) for p in ps])
    return bit_baseline, bit_coded, phase_baseline, phase_coded, ps


@app.cell
def _(bit_baseline, bit_coded, mo, phase_baseline, phase_coded, plt, ps):
    if len(ps) == 0:
        out = mo.md("_Click **Run sweep** above to generate the plot._")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(ps, bit_baseline, "o-", label="single qubit (no code)")
        axes[0].plot(ps, bit_coded, "s-", label="3-qubit bit-flip code")
        axes[0].plot(ps, ps, "k--", alpha=0.3, label="y = p")
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
        out = fig
    out
    return


if __name__ == "__main__":
    app.run()
