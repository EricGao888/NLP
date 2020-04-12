import matplotlib.pyplot as plt

idxes1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
idxes2 = [0, 1, 2, 3, 4, 5, 6, 7]
S = [6, 5, -4, -3, -2, -1, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]

A1 = []
A2 = []
S1 = []
S2 = []
for idx in range(0, 8):
    A1.append(S[idx])
    A2.append(S[idx+8])
    S1.append(min(A1[-1], A2[-1]))
    S2.append(max(A1[-1], A2[-1]))


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
plt.suptitle("Bitonic")
axes[0].plot(idxes1, S, label="S", marker='o', markersize=2)
axes[1].plot(idxes2, A1, label=r"$a_1$~$a_{\frac{n}{2}-1}$", marker='o', markersize=2)
axes[1].plot(idxes2, A2, label=r"$a_{\frac{n}{2}}$~$a_n$", marker='o', markersize=2)
axes[2].plot(idxes2, S1, label=r"$S_1$", marker='o', markersize=2)
axes[2].plot(idxes2, S2, label=r"$S_2$", marker='o', markersize=2)

for ax in axes:
    ax.set(xlabel='idx', ylabel='value')
    ax.set_xlim(min(idxes1)*1.1, max(idxes1)*1.1)
    ax.set_ylim(min(S)*1.1, max(S)*1.1)
    ax.grid()
    ax.legend()
fig.savefig("cvi.png")