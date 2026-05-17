import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator

    return (
        AerSimulator,
        ClassicalRegister,
        QuantumCircuit,
        QuantumRegister,
        mo,
        mpatches,
        plt,
        transpile,
    )


@app.cell
def _():
    data_positions = {
        0: (0.0, 2.0),
        1: (2.0, 2.0),
        2: (1.0, 1.0),
        3: (0.0, 0.0),
        4: (2.0, 0.0),
    }
    ancilla_positions = {
        0: (1.0, 2.0),
        1: (0.0, 1.0),
        2: (2.0, 1.0),
        3: (1.0, 0.0),
    }
    x_stabilizers = {
        0: [0, 1, 2],
        3: [2, 3, 4],
    }
    z_stabilizers = {
        1: [0, 2, 3],
        2: [1, 2, 4],
    }
    logical_x_qubits = [0, 3]
    logical_z_qubits = [0, 1]
    return ancilla_positions, data_positions, x_stabilizers, z_stabilizers


@app.cell
def _(
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    x_stabilizers,
    z_stabilizers,
):
    def build_x_error_circuit(error_qubits):
        data = QuantumRegister(5, "data")
        z_anc = QuantumRegister(2, "z_anc")
        syn = ClassicalRegister(2, "syn")
        qc = QuantumCircuit(data, z_anc, syn)
        qc.barrier()
        for q in error_qubits:
            qc.x(data[q])
        qc.barrier()
        z_anc_indices = sorted(z_stabilizers.keys())
        for i, a_idx in enumerate(z_anc_indices):
            for q in z_stabilizers[a_idx]:
                qc.cx(data[q], z_anc[i])
        for i in range(2):
            qc.measure(z_anc[i], syn[i])
        return qc, z_anc_indices


    def build_z_error_circuit(error_qubits):
        data = QuantumRegister(5, "data")
        x_anc = QuantumRegister(2, "x_anc")
        syn = ClassicalRegister(2, "syn")
        qc = QuantumCircuit(data, x_anc, syn)
        for i in range(5):
            qc.h(data[i])
        qc.barrier()
        for q in error_qubits:
            qc.z(data[q])
        qc.barrier()
        x_anc_indices = sorted(x_stabilizers.keys())
        for i, a_idx in enumerate(x_anc_indices):
            qc.h(x_anc[i])
            for q in x_stabilizers[a_idx]:
                qc.cx(x_anc[i], data[q])
            qc.h(x_anc[i])
        for i in range(2):
            qc.measure(x_anc[i], syn[i])
        return qc, x_anc_indices

    return build_x_error_circuit, build_z_error_circuit


@app.cell
def _(AerSimulator, transpile):
    def run_circuit(qc, shots=1024):
        sim = AerSimulator()
        tqc = transpile(qc, sim, optimization_level=0)
        result = sim.run(tqc, shots=int(shots)).result()
        return result.get_counts()


    def dominant_syndrome(counts):
        return max(counts.items(), key=lambda kv: kv[1])[0]

    return dominant_syndrome, run_circuit


@app.cell
def _():
    def parse_syndrome(bitstring, anc_indices):
        bits = bitstring.replace(" ", "")
        rev = bits[::-1]
        return {anc_indices[i]: int(rev[i]) for i in range(len(anc_indices))}


    def triggered_set(syndrome_dict):
        return {a for a, v in syndrome_dict.items() if v == 1}

    return parse_syndrome, triggered_set


@app.cell
def _(
    ancilla_positions,
    data_positions,
    mpatches,
    plt,
    x_stabilizers,
    z_stabilizers,
):
    def plot_lattice(
        triggered_ancillas=None,
        error_qubits=None,
        error_label="",
        logical_path=None,
        title="",
        save_path=None,
    ):
        triggered_ancillas = set(triggered_ancillas) if triggered_ancillas else set()
        error_qubits = set(error_qubits) if error_qubits else set()

        fig, ax = plt.subplots(figsize=(7, 7))

        for a_idx, qubits in x_stabilizers.items():
            a_pos = ancilla_positions[a_idx]
            for q in qubits:
                dp = data_positions[q]
                ax.plot(
                    [a_pos[0], dp[0]],
                    [a_pos[1], dp[1]],
                    color="#d04848",
                    linewidth=2.5,
                    alpha=0.65,
                    zorder=1,
                )
        for a_idx, qubits in z_stabilizers.items():
            a_pos = ancilla_positions[a_idx]
            for q in qubits:
                dp = data_positions[q]
                ax.plot(
                    [a_pos[0], dp[0]],
                    [a_pos[1], dp[1]],
                    color="#3060c0",
                    linewidth=2.5,
                    alpha=0.65,
                    zorder=1,
                )

        if logical_path:
            for i in range(len(logical_path) - 1):
                p1 = data_positions[logical_path[i]]
                p2 = data_positions[logical_path[i + 1]]
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    color="#20a020",
                    linewidth=4.0,
                    alpha=0.85,
                    zorder=2,
                    linestyle="--",
                )

        for d_idx, pos in data_positions.items():
            is_err = d_idx in error_qubits
            face = "#ffd060" if is_err else "white"
            circ = mpatches.Circle(
                pos,
                0.18,
                facecolor=face,
                edgecolor="black",
                linewidth=2.0,
                zorder=3,
            )
            ax.add_patch(circ)
            ax.text(
                pos[0],
                pos[1],
                f"D{d_idx}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                zorder=4,
            )
            if is_err:
                ax.text(
                    pos[0] + 0.27,
                    pos[1] + 0.27,
                    error_label,
                    fontsize=15,
                    color="#c03030",
                    fontweight="bold",
                    zorder=5,
                )

        for a_idx, pos in ancilla_positions.items():
            is_x = a_idx in x_stabilizers
            base = "#fbb" if is_x else "#bdf"
            face = "#c03030" if a_idx in triggered_ancillas else base
            text_color = "white" if a_idx in triggered_ancillas else "black"
            rect = mpatches.Rectangle(
                (pos[0] - 0.18, pos[1] - 0.18),
                0.36,
                0.36,
                facecolor=face,
                edgecolor="black",
                linewidth=2.0,
                zorder=3,
            )
            ax.add_patch(rect)
            kind = "X" if is_x else "Z"
            ax.text(
                pos[0],
                pos[1] + 0.04,
                f"A{a_idx}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
                zorder=4,
            )
            ax.text(
                pos[0],
                pos[1] - 0.07,
                f"({kind})",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
                zorder=4,
            )

        ax.set_xlim(-0.7, 2.7)
        ax.set_ylim(-0.7, 2.7)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13)

        handles = [
            mpatches.Patch(facecolor="white", edgecolor="black", label="data qubit"),
            mpatches.Patch(facecolor="#fbb", edgecolor="black", label="X-stab ancilla"),
            mpatches.Patch(facecolor="#bdf", edgecolor="black", label="Z-stab ancilla"),
            mpatches.Patch(facecolor="#c03030", edgecolor="black", label="triggered"),
            mpatches.Patch(facecolor="#ffd060", edgecolor="black", label="error site"),
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            fontsize=9,
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    return (plot_lattice,)


@app.cell
def _(plot_lattice):
    plot_lattice(
        title="[[5,1,2]] surface code lattice",
        save_path="surface_lattice.png",
    )
    return


@app.cell
def _(build_x_error_circuit):
    build_x_error_circuit([0])[0].draw("mpl", fold=-1)
    return


@app.cell
def _(build_z_error_circuit):
    build_z_error_circuit([0])[0].draw("mpl", fold=-1)
    return


@app.cell
def _(
    build_x_error_circuit,
    dominant_syndrome,
    parse_syndrome,
    plot_lattice,
    run_circuit,
    triggered_set,
):
    qc_xd0, anc_xd0 = build_x_error_circuit([0])
    counts_xd0 = run_circuit(qc_xd0, shots=2048)
    syn_xd0 = parse_syndrome(dominant_syndrome(counts_xd0), anc_xd0)
    fig_xd0 = plot_lattice(
        triggered_ancillas=triggered_set(syn_xd0),
        error_qubits={0},
        error_label="X",
        title=f"Experiment 1a: X error on D0  →  Z-syndrome {syn_xd0}",
        save_path="surface_exp1_x_on_d0.png",
    )
    fig_xd0
    return


@app.cell
def _(build_x_error_circuit):
    build_x_error_circuit([2])[0].draw("mpl", fold=-1)
    return


@app.cell
def _(
    build_x_error_circuit,
    dominant_syndrome,
    parse_syndrome,
    plot_lattice,
    run_circuit,
    triggered_set,
):
    qc_xd2, anc_xd2 = build_x_error_circuit([2])
    counts_xd2 = run_circuit(qc_xd2, shots=2048)
    print(counts_xd2)
    syn_xd2 = parse_syndrome(dominant_syndrome(counts_xd2), anc_xd2)
    fig_xd2 = plot_lattice(
        triggered_ancillas=triggered_set(syn_xd2),
        error_qubits={2},
        error_label="X",
        title=f"Experiment 1b: X error on D2 (center)  →  Z-syndrome {syn_xd2}",
        save_path="surface_exp1_x_on_d2.png",
    )
    fig_xd2
    return


@app.cell
def _(
    build_x_error_circuit,
    dominant_syndrome,
    parse_syndrome,
    plot_lattice,
    run_circuit,
    triggered_set,
):
    qc_xd4, anc_xd4 = build_x_error_circuit([4])
    counts_xd4 = run_circuit(qc_xd4, shots=2048)
    syn_xd4 = parse_syndrome(dominant_syndrome(counts_xd4), anc_xd4)
    fig_xd4 = plot_lattice(
        triggered_ancillas=triggered_set(syn_xd4),
        error_qubits={4},
        error_label="X",
        title=f"Experiment 1c: X error on D4 (corner)  →  Z-syndrome {syn_xd4}",
        save_path="surface_exp1_x_on_d4.png",
    )
    fig_xd4
    return


@app.cell
def _(
    build_z_error_circuit,
    dominant_syndrome,
    parse_syndrome,
    plot_lattice,
    run_circuit,
    triggered_set,
):
    qc_zd0, anc_zd0 = build_z_error_circuit([0])
    counts_zd0 = run_circuit(qc_zd0, shots=2048)
    syn_zd0 = parse_syndrome(dominant_syndrome(counts_zd0), anc_zd0)
    fig_zd0 = plot_lattice(
        triggered_ancillas=triggered_set(syn_zd0),
        error_qubits={0},
        error_label="Z",
        title=f"Experiment 1d: Z error on D0  →  X-syndrome {syn_zd0}",
        save_path="surface_exp1_z_on_d0.png",
    )
    fig_zd0
    return


@app.cell
def _(
    build_z_error_circuit,
    dominant_syndrome,
    parse_syndrome,
    plot_lattice,
    run_circuit,
    triggered_set,
):
    qc_zd2, anc_zd2 = build_z_error_circuit([2])
    counts_zd2 = run_circuit(qc_zd2, shots=2048)
    print(counts_zd2)
    syn_zd2 = parse_syndrome(dominant_syndrome(counts_zd2), anc_zd2)
    fig_zd2 = plot_lattice(
        triggered_ancillas=triggered_set(syn_zd2),
        error_qubits={2},
        error_label="Z",
        title=f"Experiment 1e: Z error on D2 (center)  →  X-syndrome {syn_zd2}",
        save_path="surface_exp1_z_on_d2.png",
    )
    fig_zd2
    return


@app.cell
def _(build_z_error_circuit):
    build_z_error_circuit([4])[0].draw("mpl", fold=-1 )
    return


@app.cell
def _(
    build_z_error_circuit,
    dominant_syndrome,
    parse_syndrome,
    plot_lattice,
    run_circuit,
    triggered_set,
):
    qc_zd4, anc_zd4 = build_z_error_circuit([4])
    counts_zd4 = run_circuit(qc_zd4, shots=2048)
    print(counts_zd4)
    syn_zd4 = parse_syndrome(dominant_syndrome(counts_zd4), anc_zd4)
    fig_zd4 = plot_lattice(
        triggered_ancillas=triggered_set(syn_zd4),
        error_qubits={4},
        error_label="Z",
        title=f"Experiment 1f: Z error on D4 (corner)  →  X-syndrome {syn_zd4}",
        save_path="surface_exp1_z_on_d4.png",
    )
    fig_zd4
    return


@app.cell
def _(
    build_x_error_circuit,
    build_z_error_circuit,
    dominant_syndrome,
    mo,
    parse_syndrome,
    run_circuit,
):
    def build_syndrome_table(shots=1024):
        rows = []
        for q in range(5):
            qc, ancs = build_x_error_circuit([q])
            s = parse_syndrome(dominant_syndrome(run_circuit(qc, shots=shots)), ancs)
            bits = "".join(str(s[a]) for a in sorted(s.keys()))
            triggered = ",".join(f"A{a}" for a in sorted(s) if s[a]) or "—"
            rows.append(("X", q, bits, triggered))
        for q in range(5):
            qc, ancs = build_z_error_circuit([q])
            s = parse_syndrome(dominant_syndrome(run_circuit(qc, shots=shots)), ancs)
            bits = "".join(str(s[a]) for a in sorted(s.keys()))
            triggered = ",".join(f"A{a}" for a in sorted(s) if s[a]) or "—"
            rows.append(("Z", q, bits, triggered))
        return rows


    syndrome_table_rows = build_syndrome_table()
    syndrome_table_md = "\n".join(f"| {et} on D{q} | `{bits}` | {trig} |" for et, q, bits, trig in syndrome_table_rows)
    mo.md(
        f"""
    ### Single-qubit error → syndrome table  ([[5,1,2]] surface code)

    For X errors we initialise |00000⟩ (a +1 eigenstate of every Z-stabiliser) and read the
    two Z ancillas `A1 A2`. For Z errors we initialise |+++++⟩ (a +1 eigenstate of every
    X-stabiliser) and read the two X ancillas `A0 A3`. The bitstring lists the relevant
    ancillas in increasing index order.

    | error | syndrome bits | triggered ancillas |
    |---|---|---|
    {syndrome_table_md}

    An X error on a data qubit anti-commutes with every Z-stabiliser whose support
    contains that qubit, so only those stabilisers flip. Bulk qubits (D2 here) sit in two
    plaquettes and light up two ancillas; corner qubits sit in only one. The physical
    error itself is never directly measured — only its boundary on the lattice.
    """
    )
    return


@app.cell
def _(
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    x_stabilizers,
    z_stabilizers,
):
    def build_encoded_circuit(error_type=None, error_qubits=None):
        data = QuantumRegister(5, "data")
        scratch = QuantumRegister(1, "scratch")
        ancilla = QuantumRegister(4, "ancilla")
        enc = ClassicalRegister(2, "enc")
        syn = ClassicalRegister(4, "syn")
        qc = QuantumCircuit(data, scratch, ancilla, enc, syn)

        qc.h(scratch[0])
        for q in x_stabilizers[0]:
            qc.cx(scratch[0], data[q])
        qc.h(scratch[0])
        qc.measure(scratch[0], enc[0])
        with qc.if_test((enc[0], 1)):
            qc.z(data[0])
        qc.reset(scratch[0])

        qc.h(scratch[0])
        for q in x_stabilizers[3]:
            qc.cx(scratch[0], data[q])
        qc.h(scratch[0])
        qc.measure(scratch[0], enc[1])
        with qc.if_test((enc[1], 1)):
            qc.z(data[4])
        qc.barrier()

        if error_qubits:
            for q in error_qubits:
                if error_type == "X":
                    qc.x(data[q])
                elif error_type == "Z":
                    qc.z(data[q])
                elif error_type == "Y":
                    qc.y(data[q])
        qc.barrier()

        for a_idx in sorted(x_stabilizers.keys()):
            qc.h(ancilla[a_idx])
            for q in x_stabilizers[a_idx]:
                qc.cx(ancilla[a_idx], data[q])
            qc.h(ancilla[a_idx])
        for a_idx in sorted(z_stabilizers.keys()):
            for q in z_stabilizers[a_idx]:
                qc.cx(data[q], ancilla[a_idx])
        for i in range(4):
            qc.measure(ancilla[i], syn[i])
        return qc


    def aggregate_syn(counts):
        agg = {}
        for key, n in counts.items():
            syn_bits = key.split()[0]
            agg[syn_bits] = agg.get(syn_bits, 0) + n
        return agg

    return aggregate_syn, build_encoded_circuit


@app.cell
def _(build_encoded_circuit):
    build_encoded_circuit("X", [0]).draw("mpl", fold=-1)
    return


@app.cell
def _(build_encoded_circuit):
    build_encoded_circuit("Z", [2]).draw("mpl", fold=-1)
    return


@app.cell
def _(
    aggregate_syn,
    build_encoded_circuit,
    dominant_syndrome,
    parse_syndrome,
    plot_lattice,
    run_circuit,
    triggered_set,
):
    qc_enc_x0 = build_encoded_circuit("X", [0])
    counts_enc_x0 = run_circuit(qc_enc_x0, shots=2048)
    syn_enc_x0 = parse_syndrome(dominant_syndrome(aggregate_syn(counts_enc_x0)), [0, 1, 2, 3])
    fig_enc_x0 = plot_lattice(
        triggered_ancillas=triggered_set(syn_enc_x0),
        error_qubits={0},
        error_label="X",
        title=f"Encoded |0⟩_L  +  X on D0  →  full syndrome {syn_enc_x0}",
        save_path="surface_encoded_x_on_d0.png",
    )
    fig_enc_x0
    return


@app.cell
def _(
    aggregate_syn,
    build_encoded_circuit,
    dominant_syndrome,
    parse_syndrome,
    plot_lattice,
    run_circuit,
    triggered_set,
):
    qc_enc_z0 = build_encoded_circuit("Z", [2])
    counts_enc_z0 = run_circuit(qc_enc_z0, shots=2048)
    syn_enc_z0 = parse_syndrome(dominant_syndrome(aggregate_syn(counts_enc_z0)), [0, 1, 2, 3])
    fig_enc_z0 = plot_lattice(
        triggered_ancillas=triggered_set(syn_enc_z0),
        error_qubits={2},
        error_label="Z",
        title=f"Encoded |0⟩_L  +  Z on D2  →  full syndrome {syn_enc_z0}",
        save_path="surface_encoded_z_on_d2.png",
    )
    fig_enc_z0
    return


@app.cell
def _(
    aggregate_syn,
    build_encoded_circuit,
    dominant_syndrome,
    parse_syndrome,
    plot_lattice,
    run_circuit,
    triggered_set,
):
    qc_enc_y0 = build_encoded_circuit("Y", [0])
    counts_enc_y0 = run_circuit(qc_enc_y0, shots=2048)
    syn_enc_y0 = parse_syndrome(dominant_syndrome(aggregate_syn(counts_enc_y0)), [0, 1, 2, 3])
    fig_enc_y0 = plot_lattice(
        triggered_ancillas=triggered_set(syn_enc_y0),
        error_qubits={0},
        error_label="Y",
        title=f"Encoded |0⟩_L  +  Y on D0  →  full syndrome {syn_enc_y0}",
        save_path="surface_encoded_y_on_d0.png",
    )
    fig_enc_y0
    return


@app.cell
def _(
    aggregate_syn,
    build_encoded_circuit,
    dominant_syndrome,
    parse_syndrome,
    plot_lattice,
    run_circuit,
    triggered_set,
):
    qc_enc_y2 = build_encoded_circuit("Y", [2])
    counts_enc_y2 = run_circuit(qc_enc_y2, shots=2048)
    syn_enc_y2 = parse_syndrome(dominant_syndrome(aggregate_syn(counts_enc_y2)), [0, 1, 2, 3])
    fig_enc_y2 = plot_lattice(
        triggered_ancillas=triggered_set(syn_enc_y2),
        error_qubits={2},
        error_label="Y",
        title=f"Encoded |0⟩_L  +  Y on D2 (center)  →  full syndrome {syn_enc_y2}",
        save_path="surface_encoded_y_on_d2.png",
    )
    fig_enc_y2
    return


@app.cell
def _(
    aggregate_syn,
    build_encoded_circuit,
    dominant_syndrome,
    mo,
    parse_syndrome,
    run_circuit,
):
    def build_encoded_syndrome_table(shots=1024):
        rows = []
        for et in ("X", "Z", "Y"):
            for q in range(5):
                qc = build_encoded_circuit(et, [q])
                s = parse_syndrome(
                    dominant_syndrome(aggregate_syn(run_circuit(qc, shots=shots))),
                    [0, 1, 2, 3],
                )
                bits = "".join(str(s[a]) for a in sorted(s.keys()))
                triggered = ",".join(f"A{a}" for a in sorted(s) if s[a]) or "—"
                rows.append((et, q, bits, triggered))
        return rows


    encoded_table_rows = build_encoded_syndrome_table()
    encoded_table_md = "\n".join(f"| {et} on D{q} | `{bits}` | {trig} |" for et, q, bits, trig in encoded_table_rows)
    mo.md(
        f"""
    ### Full syndrome table on the true |0⟩_L codeword

    The state is first projected onto the joint +1 eigenspace of both X-stabilisers via
    mid-circuit measurement of a scratch ancilla and a conditional Z correction
    (`Z_D0` for A0, `Z_D4` for A3). After that all four stabilisers commute with the
    state and measure +1 deterministically, so any single Pauli error produces a
    repeatable 4-bit syndrome on `A3 A2 A1 A0`.

    | error | A3 A2 A1 A0 | triggered ancillas |
    |---|---|---|
    {encoded_table_md}

    A Y error is X·Z up to phase, so its syndrome is the XOR of the X and Z syndromes on
    the same qubit — both halves of the code light up at once. The center qubit D2 sits
    in every stabiliser, so `Y on D2` triggers all four ancillas.
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
