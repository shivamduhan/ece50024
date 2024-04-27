import csv
import matplotlib.pyplot as plt


def plot_loss_epoch(filename):
    with open(filename, "r") as f:
        data = csv.reader(f)
        header = next(data)
        epoch = []
        loss = []
        for row in data:
            epoch.append(int(row[0]))
            loss.append(float(row[1]))

        plt.plot(epoch, loss)


if __name__ == "__main__":
    plot_loss_epoch("loss_log_300ml.csv")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.savefig("pictures/loss_log/300ml.png")
    plt.show()
