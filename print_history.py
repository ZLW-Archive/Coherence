import matplotlib.pyplot as plt
import pickle

CHECKPOINT = "good"
PATH = "checkpoint/{}/trainHistory".format(CHECKPOINT)
PIC = "checkpoint/{}/pic".format(CHECKPOINT)

file = open(PATH, "rb")
history = pickle.load(file)

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig(PIC)
plt.show()

