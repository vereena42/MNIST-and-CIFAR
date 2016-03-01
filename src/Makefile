CURL ?= curl
EXTRACT_FILE_FROM_TAR_GZ ?= tar --transform='s/.*\///' -xzvmf

MNIST_URL = http://yann.lecun.com/exdb/mnist

MNIST_DATASETS =
MNIST_DATASETS += train-images-idx3-ubyte.gz
MNIST_DATASETS += t10k-images-idx3-ubyte.gz
MNIST_DATASETS += train-labels-idx1-ubyte.gz
MNIST_DATASETS += t10k-labels-idx1-ubyte.gz

CIFAR_URL = https://www.cs.toronto.edu/~kriz

CIFAR_ARCHIVE = cifar-10-python.tar.gz
CIFAR_ARCHIVE_PREFIX = cifar-10-batches-py

CIFAR_DATASETS =
CIFAR_DATASETS += test_batch
CIFAR_DATASETS += data_batch_1
CIFAR_DATASETS += data_batch_2
CIFAR_DATASETS += data_batch_3
CIFAR_DATASETS += data_batch_4
CIFAR_DATASETS += data_batch_5
CIFAR_DATASETS += batches.meta

all: mnist cifar

mnist: $(MNIST_DATASETS)

$(MNIST_DATASETS):
	$(CURL) $(MNIST_URL)/$@ > $@

cifar: $(CIFAR_DATASETS)

$(CIFAR_DATASETS): $(CIFAR_ARCHIVE)
	$(EXTRACT_FILE_FROM_TAR_GZ) $< $(CIFAR_ARCHIVE_PREFIX)/$@

$(CIFAR_ARCHIVE):
	$(CURL) $(CIFAR_URL)/$@ > $@

clean:
	$(RM) $(MNIST_DATASETS) $(CIFAR_DATASETS) $(CIFAR_ARCHIVE)

.PHONY: all clean mnist cifar
