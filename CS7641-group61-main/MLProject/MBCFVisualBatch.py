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

show_range = 50
epochs = list(range(1, show_range + 1))
val_steps = list(range(0, show_range + 5, 5))

bces_b4096, HR5_b4096, HR10_b4096, HR20_b4096, NDCG5_b4096, NDCG10_b4096, NDCG20_b4096 = get_result("./MBCF_b4096.log", show_range)
bces_b2048, HR5_b2048, HR10_b2048, HR20_b2048, NDCG5_b2048, NDCG10_b2048, NDCG20_b2048 = get_result("./MBCF_b2048.log", show_range)
bces_b1024, HR5_b1024, HR10_b1024, HR20_b1024, NDCG5_b1024, NDCG10_b1024, NDCG20_b1024 = get_result("./MBCF_b1024.log", show_range)

plt.figure(figsize=(18, 6))

# --- BCE ---
plt.subplot(1, 3, 1)
plt.plot(epochs, bces_b4096, label="Batch 4096")
plt.plot(epochs, bces_b2048, label="Batch 2048")
plt.plot(epochs, bces_b1024, label="Batch 1024")
plt.xlabel("Epoch")
plt.ylabel("Training BCE Loss")
plt.title("Training BCE over Epochs")
plt.grid(True)
plt.legend()

# --- HR Metrics ---
plt.subplot(1, 3, 2)
plt.plot(val_steps, HR20_b4096, label="HR@20 Batch 4096")
plt.plot(val_steps, HR20_b2048, label="HR@20 Batch 2048")
plt.plot(val_steps, HR20_b1024, label="HR@20 Batch 1024")
plt.plot(val_steps, HR10_b4096, label="HR@10 Batch 4096")
plt.plot(val_steps, HR10_b2048, label="HR@10 Batch 2048")
plt.plot(val_steps, HR10_b1024, label="HR@10 Batch 1024")
plt.plot(val_steps, HR5_b4096, label="HR@5 Batch 4096")
plt.plot(val_steps, HR5_b2048, label="HR@5 Batch 2048")
plt.plot(val_steps, HR5_b1024, label="HR@5 Batch 1024")
plt.xlabel("Evaluation Step")
plt.ylabel("HR Score")
plt.title("HR Metrics Over Training")
plt.grid(True)
plt.legend()

# --- NDCG Metrics ---
plt.subplot(1, 3, 3)
plt.plot(val_steps, NDCG20_b4096, label="NDCG@20 Batch 4096")
plt.plot(val_steps, NDCG20_b2048, label="NDCG@20 Batch 2048")
plt.plot(val_steps, NDCG20_b1024, label="NDCG@20 Batch 1024")
plt.plot(val_steps, NDCG10_b4096, label="NDCG@10 Batch 4096")
plt.plot(val_steps, NDCG10_b2048, label="NDCG@10 Batch 2048")
plt.plot(val_steps, NDCG10_b1024, label="NDCG@10 Batch 1024")
plt.plot(val_steps, NDCG5_b4096, label="NDCG@5 Batch 4096")
plt.plot(val_steps, NDCG5_b2048, label="NDCG@5 Batch 2048")
plt.plot(val_steps, NDCG5_b1024, label="NDCG@5 Batch 1024")
plt.xlabel("Evaluation Step")
plt.ylabel("NDCG Score")
plt.title("NDCG Metrics Over Training")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()