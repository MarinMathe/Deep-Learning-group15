import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ==============================================================================
# 1.  DATA LOADING & PREPROCESSING
# ==============================================================================


def LoadBookData(book_path="goblet_book.txt"):
    """Read the book and build char<->index mappings."""
    with open(book_path, "r", encoding="utf-8") as fid:
        book_data = fid.read()

    unique_chars = list(set(book_data))
    K = len(unique_chars)

    char_to_ind = {ch: i for i, ch in enumerate(unique_chars)}
    ind_to_char = {i: ch for i, ch in enumerate(unique_chars)}

    print(f"[Data] Total chars: {len(book_data):,}   Unique chars: {K}")
    return book_data, unique_chars, char_to_ind, ind_to_char, K


def CharsToOneHot(chars, char_to_ind, K):
    """Convert a string of characters to a (K, T) one-hot matrix."""
    T = len(chars)
    X = np.zeros((K, T), dtype=np.float64)
    for t, ch in enumerate(chars):
        X[char_to_ind[ch], t] = 1.0
    return X


def OneHotToChars(Y, ind_to_char):
    """Convert a (K, T) one-hot matrix back to a string."""
    indices = np.argmax(Y, axis=0)
    return "".join(ind_to_char[i] for i in indices)


# ==============================================================================
# 2.  RNN INITIALISATION
# ==============================================================================


def InitRNN(K, m=100, seed=42):
    """
    Initialise RNN parameters.

    Parameters
    ----------
    K : vocabulary size (input / output dim)
    m : hidden state dimension
    """
    rng = np.random.default_rng(seed)
    RNN = {
        "b": np.zeros((m, 1), dtype=np.float64),  # hidden bias
        "c": np.zeros((K, 1), dtype=np.float64),  # output bias
        "U": (1 / np.sqrt(2 * K)) * rng.standard_normal((m, K)),  # input weights
        "W": (1 / np.sqrt(2 * m)) * rng.standard_normal((m, m)),  # recurrent weights
        "V": (1 / np.sqrt(m)) * rng.standard_normal((K, m)),  # output weights
    }
    return RNN


# ==============================================================================
# 3.  SOFTMAX
# ==============================================================================


def Softmax(o):
    """Numerically stable softmax along axis 0 (column-wise)."""
    o = o - np.max(o, axis=0, keepdims=True)
    e = np.exp(o)
    return e / np.sum(e, axis=0, keepdims=True)


# ==============================================================================
# 4.  SYNTHESIZE TEXT
# ==============================================================================


def SynthesizeText(RNN, h0, x0, n, rng=None):
    """
    Generate a sequence of n characters from the RNN.

    Parameters
    ----------
    RNN  : parameter dict
    h0   : (m, 1) initial hidden state
    x0   : (K, 1) first input (one-hot)
    n    : number of characters to generate

    Returns
    -------
    Y    : (K, n) one-hot encoded generated sequence
    """
    if rng is None:
        rng = np.random.default_rng()

    K = RNN["V"].shape[0]
    m = RNN["W"].shape[0]

    Y = np.zeros((K, n), dtype=np.float32)
    h = h0.copy()
    x = x0.copy()

    for t in range(n):
        a = RNN["W"] @ h + RNN["U"] @ x + RNN["b"]
        h = np.tanh(a)
        o = RNN["V"] @ h + RNN["c"]
        p = Softmax(o)

        # Sample from the discrete distribution
        cp = np.cumsum(p, axis=0)
        draw = rng.uniform()
        ii = int(np.argmax(cp - draw > 0))

        Y[ii, t] = 1.0
        x = np.zeros((K, 1), dtype=np.float64)
        x[ii, 0] = 1.0

    return Y


# ==============================================================================
# 5.  FORWARD PASS
# ==============================================================================


def ForwardPass(X, Y, RNN, h0):
    """
    Forward pass of the vanilla RNN.

    Parameters
    ----------
    X   : (K, T) input one-hot matrix
    Y   : (K, T) target one-hot matrix
    RNN : parameter dict
    h0  : (m, 1) initial hidden state

    Returns
    -------
    loss  : scalar (average cross-entropy)
    cache : dict of intermediates for backward pass
    """
    K, T = X.shape
    m = RNN["W"].shape[0]

    a_seq = np.zeros((m, T), dtype=np.float64)
    h_seq = np.zeros(
        (m, T + 1), dtype=np.float64
    )  # h_seq[:, 0] = h0, h_seq[:, t] = h_t
    o_seq = np.zeros((K, T), dtype=np.float64)
    p_seq = np.zeros((K, T), dtype=np.float64)

    h_seq[:, 0:1] = h0

    for t in range(T):
        a_seq[:, t : t + 1] = (
            RNN["W"] @ h_seq[:, t : t + 1] + RNN["U"] @ X[:, t : t + 1] + RNN["b"]
        )
        h_seq[:, t + 1 : t + 2] = np.tanh(a_seq[:, t : t + 1])
        o_seq[:, t : t + 1] = RNN["V"] @ h_seq[:, t + 1 : t + 2] + RNN["c"]
        p_seq[:, t : t + 1] = Softmax(o_seq[:, t : t + 1])

    loss = -np.mean(np.sum(Y * np.log(np.maximum(p_seq, 1e-15)), axis=0))

    cache = {
        "X": X,
        "Y": Y,
        "a_seq": a_seq,
        "h_seq": h_seq,
        "p_seq": p_seq,
    }
    return loss, cache


# ==============================================================================
# 6.  BACKWARD PASS
# ==============================================================================


def BackwardPass(RNN, cache):
    """
    Backward pass for the vanilla RNN (BPTT).

    Returns
    -------
    grads : dict with keys b, c, U, W, V
    h_last: last hidden state
    """
    X = cache["X"]
    Y = cache["Y"]
    a_seq = cache["a_seq"]
    h_seq = cache["h_seq"]
    p_seq = cache["p_seq"]

    K, T = Y.shape
    m = RNN["W"].shape[0]

    grad_V = np.zeros_like(RNN["V"])
    grad_c = np.zeros_like(RNN["c"])
    grad_W = np.zeros_like(RNN["W"])
    grad_U = np.zeros_like(RNN["U"])
    grad_b = np.zeros_like(RNN["b"])

    # Gradient flowing into h at time T+1 (zero at the end)
    dh_next = np.zeros((m, 1), dtype=np.float64)

    for t in reversed(range(T)):
        # Gradient of loss w.r.t. output logits
        g_o = -(Y[:, t : t + 1] - p_seq[:, t : t + 1])  # (K, 1)

        # Gradients for V and c
        grad_V += g_o @ h_seq[:, t + 1 : t + 2].T  # (K, m)
        grad_c += g_o  # (K, 1)

        # Backprop into h_t
        dh = RNN["V"].T @ g_o + dh_next  # (m, 1)

        # Backprop through tanh
        da = dh * (1 - np.tanh(a_seq[:, t : t + 1]) ** 2)  # (m, 1)

        # Gradients for W, U, b
        grad_W += da @ h_seq[:, t : t + 1].T  # (m, m)
        grad_U += da @ X[:, t : t + 1].T  # (m, K)
        grad_b += da  # (m, 1)

        dh_next = RNN["W"].T @ da

    # Divide by T (matching average loss)
    grads = {
        "V": grad_V / T,
        "c": grad_c / T,
        "W": grad_W / T,
        "U": grad_U / T,
        "b": grad_b / T,
    }

    # Clip gradients to avoid explosion
    for key in grads:
        np.clip(grads[key], -5, 5, out=grads[key])

    h_last = h_seq[:, T : T + 1]
    return grads, h_last


# ==============================================================================
# 7.  GRADIENT CHECK  (vs PyTorch)
# ==============================================================================


def ComputeGradsWithTorch(X, y, h0, RNN):
    """
    Professor's skeleton (column-wise storage: X is K x tau, h0 is m x 1).

    Filled-in equations (1) and (2):
        a_t = W * h_{t-1} + U * x_t + b
        h_t = tanh(a_t)

    Note: RNN keys must use the shapes below
        W  : (m, m)
        U  : (m, K)
        V  : (K, m)
        b  : (m, 1)
        c  : (K, 1)
    """
    tau = X.shape[1]

    Xt = torch.from_numpy(X)  # (K, tau)
    ht = torch.from_numpy(h0)  # (m, 1)

    torch_network = {}
    for kk in RNN.keys():
        torch_network[kk] = torch.tensor(RNN[kk], requires_grad=True)

    # Named torch activation classes (as in the skeleton)
    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=0)

    # Storage for hidden states: (m, tau)
    Hs = torch.empty(h0.shape[0], X.shape[1], dtype=torch.float64)

    hprev = ht
    for t in range(tau):

        #### BEGIN filled-in code ######
        # Equations (1) and (2):
        #   a_t = W h_{t-1} + U x_t + b
        #   h_t = tanh(a_t)
        a_t = (
            torch.matmul(torch_network["W"], hprev)
            + torch.matmul(torch_network["U"], Xt[:, t : t + 1])
            + torch_network["b"]
        )  # (m, 1)
        hprev = apply_tanh(a_t)  # (m, 1)
        Hs[:, t : t + 1] = hprev
        #### END of filled-in code ######

    # Equations (3) and (4) – provided by the skeleton, untouched
    Os = torch.matmul(torch_network["V"], Hs) + torch_network["c"]  # (K, tau)
    P = apply_softmax(Os)  # (K, tau)

    # Loss – provided by the skeleton (column-wise version)
    loss = torch.mean(-torch.log(P[y, np.arange(tau)]))

    # Backward pass and gradient extraction – provided by the skeleton
    loss.backward()

    grads = {}
    for kk in RNN.keys():
        grads[kk] = torch_network[kk].grad.numpy()

    return grads


def GradientCheck(book_data, char_to_ind, ind_to_char, K, m_check=10, seq_length=25):
    """
    Compare analytic gradients (BackwardPass) against PyTorch autograd.

    """
    print("\n" + "=" * 55)
    print("GRADIENT CHECK  (m=10, seq_length=25)")
    print("=" * 55)

    RNN_check = InitRNN(K, m=m_check, seed=1)

    # Build inputs: first seq_length chars of the book
    X_chars = book_data[:seq_length]
    Y_chars = book_data[1 : seq_length + 1]
    X_np = CharsToOneHot(X_chars, char_to_ind, K)  # (K, seq_length)
    Y_np = CharsToOneHot(Y_chars, char_to_ind, K)  # (K, seq_length)

    # Integer labels needed by the torch loss line  P[y, np.arange(tau)]
    y_int = np.array([char_to_ind[ch] for ch in Y_chars])  # (seq_length,)

    h0 = np.zeros((m_check, 1), dtype=np.float64)

    # Analytic gradients
    _, cache = ForwardPass(X_np, Y_np, RNN_check, h0)
    my_grads, _ = BackwardPass(RNN_check, cache)

    # PyTorch reference gradients (professor's skeleton, filled in above)
    ref_grads = ComputeGradsWithTorch(X_np, y_int, h0, RNN_check)

    def rel_error(a, b):
        return np.max(np.abs(a - b) / np.maximum(1e-10, np.abs(a) + np.abs(b)))

    all_ok = True
    for key in ["W", "U", "V", "b", "c"]:
        err = rel_error(my_grads[key], ref_grads[key])
        ok = err < 1e-5
        all_ok = all_ok and ok
        status = "✅" if ok else "❌"
        print(f"  {key}   max_rel_err = {err:.2e}  {status}")

    print("=" * 55)
    if all_ok:
        print("All gradients look correct ✅")
    else:
        print("Some gradients have large errors ❌ – check BackwardPass")
    return all_ok


# ==============================================================================
# 8.  ADAM OPTIMIZER
# ==============================================================================


def InitAdam(RNN, beta1=0.9, beta2=0.999, eps=1e-8):
    """Initialise Adam state."""
    adam = {
        "m": {k: np.zeros_like(v) for k, v in RNN.items()},
        "v": {k: np.zeros_like(v) for k, v in RNN.items()},
        "t": 0,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
    }
    return adam


def AdamUpdate(RNN, grads, adam, eta):
    """Apply one Adam update step in-place."""
    adam["t"] += 1
    t0 = adam["t"]
    b1, b2, eps = adam["beta1"], adam["beta2"], adam["eps"]

    for key in RNN:
        g = grads[key]
        adam["m"][key] = b1 * adam["m"][key] + (1 - b1) * g
        adam["v"][key] = b2 * adam["v"][key] + (1 - b2) * g**2

        m_hat = adam["m"][key] / (1 - b1**t0)
        v_hat = adam["v"][key] / (1 - b2**t0)

        RNN[key] -= eta / (np.sqrt(v_hat) + eps) * m_hat


# ==============================================================================
# 9.  TRAINING LOOP
# ==============================================================================


def TrainRNN(
    book_data,
    char_to_ind,
    ind_to_char,
    K,
    m=100,
    eta=0.001,
    seq_length=25,
    n_epochs=3,
    synth_every=1000,
    synth_len=200,
    print_every=100,
    save_best=True,
):
    """
    Train the RNN with Adam for n_epochs over book_data.

    Returns
    -------
    RNN          : trained parameters
    best_RNN     : parameters at lowest smooth_loss checkpoint
    smooth_losses: list of (iteration, smooth_loss) tuples
    synth_samples: list of (iteration, text) tuples
    """
    rng = np.random.default_rng(42)
    RNN = InitRNN(K, m=m, seed=42)
    adam = InitAdam(RNN)

    N = len(book_data)
    smooth_loss = None
    best_loss = np.inf
    best_RNN = None

    smooth_losses = []
    synth_samples = []

    iteration = 0
    start_time = time.time()

    for epoch in range(n_epochs):
        e = 0
        hprev = np.zeros((m, 1), dtype=np.float64)

        while e < N - seq_length - 1:
            X_chars = book_data[e : e + seq_length]
            Y_chars = book_data[e + 1 : e + seq_length + 1]

            X = CharsToOneHot(X_chars, char_to_ind, K)
            Y = CharsToOneHot(Y_chars, char_to_ind, K)

            # Forward + backward
            loss, cache = ForwardPass(X, Y, RNN, hprev)
            grads, hprev = BackwardPass(RNN, cache)

            # Adam update
            AdamUpdate(RNN, grads, adam, eta)

            # Smooth loss
            if smooth_loss is None:
                smooth_loss = loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss

            # Synthesize text periodically
            if iteration % synth_every == 0:
                x0 = X[:, 0:1].copy()
                Y_synth = SynthesizeText(RNN, hprev, x0, synth_len, rng)
                text = OneHotToChars(Y_synth, ind_to_char)
                synth_samples.append((iteration, smooth_loss, text))
                print(f"\n--- iter={iteration}, smooth_loss={smooth_loss:.6f} ---")
                print(text)
                print()

            # Print smooth loss
            if iteration % print_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"  epoch={epoch+1} iter={iteration:7d}  "
                    f"smooth_loss={smooth_loss:.6f}  ({elapsed:.0f}s)"
                )

            smooth_losses.append((iteration, smooth_loss))

            # Track best model
            if save_best and smooth_loss < best_loss:
                best_loss = smooth_loss
                best_RNN = {k: v.copy() for k, v in RNN.items()}

            e += seq_length
            iteration += 1

        # End of epoch: reset hidden state
        hprev = np.zeros((m, 1), dtype=np.float64)
        print(f"\n[Epoch {epoch+1}/{n_epochs} done]  smooth_loss={smooth_loss:.6f}\n")

    return RNN, best_RNN, smooth_losses, synth_samples


# ==============================================================================
# 10.  PLOTTING
# ==============================================================================


def PlotSmoothLoss(smooth_losses, save_path="smooth_loss.png"):
    iters, losses = zip(*smooth_losses)
    plt.figure(figsize=(12, 5))
    plt.plot(iters, losses, linewidth=0.8, color="steelblue")
    plt.xlabel("Update step")
    plt.ylabel("Smooth loss")
    plt.title("Smooth loss during training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Plot] Saved {save_path}")
    plt.close()


def SaveSynthSamples(synth_samples, save_path="synth_samples.txt"):
    """Save all synthesized text samples to a file for the report."""
    with open(save_path, "w", encoding="utf-8") as f:
        for it, sl, text in synth_samples:
            f.write(f"{'='*60}\n")
            f.write(f"iter={it:7d}   smooth_loss={sl:.6f}\n")
            f.write(f"{'='*60}\n")
            f.write(text + "\n\n")
    print(f"[Synth] Saved {save_path}")


def GenerateBestPassage(
    best_RNN,
    K,
    ind_to_char,
    char_to_ind,
    first_char=".",
    length=1000,
    save_path="best_passage.txt",
):
    """Generate a 1000-char passage from the best model."""
    rng = np.random.default_rng(99)
    m = best_RNN["W"].shape[0]
    h0 = np.zeros((m, 1), dtype=np.float64)
    x0 = np.zeros((K, 1), dtype=np.float64)
    x0[char_to_ind[first_char], 0] = 1.0

    Y = SynthesizeText(best_RNN, h0, x0, length, rng)
    text = OneHotToChars(Y, ind_to_char)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Best] Saved 1000-char passage to {save_path}")
    print(text)
    return text


# ==============================================================================
# 11.  MAIN
# ==============================================================================

if __name__ == "__main__":

    BOOK_PATH = "goblet_book.txt"  # <-- put the book file here
    M = 100  # hidden state size
    ETA = 0.001  # learning rate
    SEQ_LENGTH = 25  # sequence length for training
    N_EPOCHS = 3  # number of full passes over the book
    SYNTH_EVERY = 1000  # synthesize text every N update steps
    PRINT_EVERY = 100  # print smooth loss every N steps

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    book_data, unique_chars, char_to_ind, ind_to_char, K = LoadBookData(BOOK_PATH)

    # ------------------------------------------------------------------
    # 2. Gradient check  (uses small network m=10)
    # ------------------------------------------------------------------
    GradientCheck(book_data, char_to_ind, ind_to_char, K, m_check=10, seq_length=25)

    # ------------------------------------------------------------------
    # 3. Synthesize from random init (before training)
    # ------------------------------------------------------------------
    print("\n--- Synthesis BEFORE training (random init) ---")
    RNN_init = InitRNN(K, m=M, seed=42)
    rng_pre = np.random.default_rng(0)
    h0_pre = np.zeros((M, 1), dtype=np.float64)
    x0_pre = np.zeros((K, 1), dtype=np.float64)
    x0_pre[char_to_ind["."], 0] = 1.0
    Y_pre = SynthesizeText(RNN_init, h0_pre, x0_pre, 200, rng_pre)
    print(OneHotToChars(Y_pre, ind_to_char))

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Training RNN  m={M}  eta={ETA}  seq_length={SEQ_LENGTH}")
    print("=" * 60)

    RNN, best_RNN, smooth_losses, synth_samples = TrainRNN(
        book_data,
        char_to_ind,
        ind_to_char,
        K,
        m=M,
        eta=ETA,
        seq_length=SEQ_LENGTH,
        n_epochs=N_EPOCHS,
        synth_every=SYNTH_EVERY,
        synth_len=200,
        print_every=PRINT_EVERY,
    )

    # ------------------------------------------------------------------
    # 5. Plot smooth loss
    # ------------------------------------------------------------------
    PlotSmoothLoss(smooth_losses, save_path="smooth_loss.png")

    # ------------------------------------------------------------------
    # 6. Save synthesis samples
    # ------------------------------------------------------------------
    SaveSynthSamples(synth_samples, save_path="synth_samples.txt")

    # ------------------------------------------------------------------
    # 7. Generate 1000-char passage from best model
    # ------------------------------------------------------------------
    GenerateBestPassage(
        best_RNN,
        K,
        ind_to_char,
        char_to_ind,
        first_char=".",
        length=1000,
        save_path="best_passage.txt",
    )

    print("\n✅ Assignment 4 complete.")
