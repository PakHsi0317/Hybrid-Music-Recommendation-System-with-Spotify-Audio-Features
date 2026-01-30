import re
import matplotlib.pyplot as plt

def get_result(log_path="./MBCF.log", show_range = 50):
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

show_range = 100
epochs = list(range(1, show_range + 1))
val_steps = list(range(0, show_range + 5, 5))

bces_lr005, HR5_lr005, HR10_lr005, HR20_lr005, NDCG5_lr005, NDCG10_lr005, NDCG20_lr005 = get_result("./MBCF_lr005.log", show_range)
bces_lr001, HR5_lr001, HR10_lr001, HR20_lr001, NDCG5_lr001, NDCG10_lr001, NDCG20_lr001 = get_result("./MBCF_lr001.log", show_range)
bces_lr0005, HR5_lr0005, HR10_lr0005, HR20_lr0005, NDCG5_lr0005, NDCG10_lr0005, NDCG20_lr0005 = get_result("./MBCF_lr0005.log", show_range)
bces_lr0001, HR5_lr0001, HR10_lr0001, HR20_lr0001, NDCG5_lr0001, NDCG10_lr0001, NDCG20_lr0001 = get_result("./MBCF_lr0001.log", show_range)

plt.figure(figsize=(18, 6))

# --- BCE ---
plt.subplot(1, 3, 1)
plt.plot(epochs, bces_lr005, label="LR 0.005")
plt.plot(epochs, bces_lr001, label="LR 0.001")
plt.plot(epochs, bces_lr0005, label="LR 0.0005")
plt.plot(epochs, bces_lr0001, label="LR 0.0001")
plt.xlabel("Epoch")
plt.ylabel("Training BCE Loss")
plt.title("Training BCE over Epochs")
plt.grid(True)
plt.legend()

# --- HR Metrics ---
plt.subplot(1, 3, 2)
plt.plot(val_steps, HR20_lr005, label="HR@20 LR 0.005")
plt.plot(val_steps, HR20_lr001, label="HR@20 LR 0.001")
plt.plot(val_steps, HR20_lr0005, label="HR@20 LR 0.0005")
plt.plot(val_steps, HR20_lr0001, label="HR@20 LR 0.0001")
plt.plot(val_steps, HR10_lr005, label="HR@10 LR 0.005")
plt.plot(val_steps, HR10_lr001, label="HR@10 LR 0.001")
plt.plot(val_steps, HR10_lr0005, label="HR@10 LR 0.0005")
plt.plot(val_steps, HR10_lr0001, label="HR@10 LR 0.0001")
plt.plot(val_steps, HR5_lr005, label="HR@5 LR 0.005")
plt.plot(val_steps, HR5_lr001, label="HR@5 LR 0.001")
plt.plot(val_steps, HR5_lr0005, label="HR@5 LR 0.0005")
plt.plot(val_steps, HR5_lr0001, label="HR@5 LR 0.0001")
plt.xlabel("Evaluation Step")
plt.ylabel("HR Score")
plt.title("HR Metrics Over Training")
plt.grid(True)
plt.legend()

# --- NDCG Metrics ---
plt.subplot(1, 3, 3)
plt.plot(val_steps, NDCG20_lr005, label="NDCG@20 LR 0.005")
plt.plot(val_steps, NDCG20_lr001, label="NDCG@20 LR 0.001")
plt.plot(val_steps, NDCG20_lr0005, label="NDCG@20 LR 0.0005")
plt.plot(val_steps, NDCG20_lr0001, label="NDCG@20 LR 0.0001")
plt.plot(val_steps, NDCG10_lr005, label="NDCG@10 LR 0.005")
plt.plot(val_steps, NDCG10_lr001, label="NDCG@10 LR 0.001")
plt.plot(val_steps, NDCG10_lr0005, label="NDCG@10 LR 0.0005")
plt.plot(val_steps, NDCG10_lr0001, label="NDCG@10 LR 0.0001")
plt.plot(val_steps, NDCG5_lr005, label="NDCG@5 LR 0.005")
plt.plot(val_steps, NDCG5_lr001, label="NDCG@5 LR 0.001")
plt.plot(val_steps, NDCG5_lr0005, label="NDCG@5 LR 0.0005")
plt.plot(val_steps, NDCG5_lr0001, label="NDCG@5 LR 0.0001")
plt.xlabel("Evaluation Step")
plt.ylabel("NDCG Score")
plt.title("NDCG Metrics Over Training")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()