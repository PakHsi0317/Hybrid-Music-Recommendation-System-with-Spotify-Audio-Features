import re
import matplotlib.pyplot as plt

def get_result(log_path="./MBCF_withneg.log", show_range = 50):
    with open(log_path, "r") as f:
        log = f.read()

    epoch_bce = []
    for m in re.finditer(r"Epoch\s+(\d+)\s*\|\s*training BCE\s+([0-9.]+)", log):
        epoch = int(m.group(1))
        bce = float(m.group(2))
        epoch_bce.append((epoch, bce))

    bces = [b for _, b in epoch_bce][:show_range]

    pattern = (
        r"Test: \{'HR@5': ([0-9.e-]+), 'NDCG@5': ([0-9.e-]+), "
        r"'HR@10': ([0-9.e-]+), 'NDCG@10': ([0-9.e-]+), "
        r"'HR@20': ([0-9.e-]+), 'NDCG@20': ([0-9.e-]+)\}"
    )

    matches = list(re.finditer(pattern, log))

    val_range = (int) (show_range / 5) + 1
    HR5 = [float(m.group(1)) for m in matches][:val_range]
    HR10 = [float(m.group(3)) for m in matches][:val_range]
    HR20 = [float(m.group(5)) for m in matches][:val_range]

    NDCG5 = [float(m.group(2)) for m in matches][:val_range]
    NDCG10 = [float(m.group(4)) for m in matches][:val_range]
    NDCG20 = [float(m.group(6)) for m in matches][:val_range]

    return bces, HR5, HR10, HR20, NDCG5, NDCG10, NDCG20

show_range = 50
epochs = list(range(1, show_range + 1))
val_steps = list(range(0, show_range + 5, 5))

bces_withneg, HR5_withneg, HR10_withneg, HR20_withneg, NDCG5_withneg, NDCG10_withneg, NDCG20_withneg = get_result("./MBCF_withneg.log", show_range)
bces_noneg, HR5_noneg, HR10_noneg, HR20_noneg, NDCG5_noneg, NDCG10_noneg, NDCG20_noneg = get_result("./MBCF_noneg.log", show_range)

plt.figure(figsize=(18, 6))

# --- BCE ---
plt.subplot(1, 3, 1)
plt.plot(epochs, bces_withneg, label="with neg")
plt.plot(epochs, bces_noneg, label="no neg")
plt.xlabel("Epoch")
plt.ylabel("Training BCE Loss")
plt.title("Training BCE over Epochs")
plt.grid(True)
plt.legend()

# --- HR Metrics ---
plt.subplot(1, 3, 2)
plt.plot(val_steps, HR20_withneg, label="HR@20 with neg")
plt.plot(val_steps, HR20_noneg, label="HR@20 no neg")
plt.plot(val_steps, HR10_withneg, label="HR@10 with neg")
plt.plot(val_steps, HR10_noneg, label="HR@10 no neg")
plt.plot(val_steps, HR5_withneg, label="HR@5 with neg")
plt.plot(val_steps, HR5_noneg, label="HR@5 no neg")
plt.xlabel("Evaluation Step")
plt.ylabel("HR Score")
plt.title("HR Metrics Over Training")
plt.grid(True)
plt.legend()

# --- NDCG Metrics ---
plt.subplot(1, 3, 3)
plt.plot(val_steps, NDCG20_withneg, label="NDCG@20 with neg")
plt.plot(val_steps, NDCG20_noneg, label="NDCG@20 no neg")
plt.plot(val_steps, NDCG10_withneg, label="NDCG@10 with neg")
plt.plot(val_steps, NDCG10_noneg, label="NDCG@10 no neg")
plt.plot(val_steps, NDCG5_withneg, label="NDCG@5 with neg")
plt.plot(val_steps, NDCG5_noneg, label="NDCG@5 no neg")
plt.xlabel("Evaluation Step")
plt.ylabel("NDCG Score")
plt.title("NDCG Metrics Over Training")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()