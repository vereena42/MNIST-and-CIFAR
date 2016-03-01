# Copyright (c) 2016, Dominika Salawa <vereena42@gmail.com>
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
# 
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
# 
#     * Neither the name of the <organization> nor the names of its
#       contributors may be used to endorse or promote products derived from this
#       software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import gzip
import numpy
import os
import matplotlib
import matplotlib.pyplot as plt
from neural_network import NeuralNetworkModel

num_of_epochs=25
dtype = numpy.float64

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            print ("Problem z otwarciem obrazkow!")
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows * cols)
        return data/255.

def extract_labels(filename):
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            print ("Problem z otwarciem labelek!")
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        return labels

train_images = extract_images(os.path.join("", 'train-images-idx3-ubyte.gz'))
test_images = extract_images(os.path.join("", 't10k-images-idx3-ubyte.gz'))
train_labels = extract_labels(os.path.join("", 'train-labels-idx1-ubyte.gz'))
test_labels = extract_labels(os.path.join("", 't10k-labels-idx1-ubyte.gz'))
model=NeuralNetworkModel(False,[500],784,60000,train_images,train_labels,10000,test_images,test_labels,True)


i=0
while i<num_of_epochs:
    model.train()
    print i+1,model.validate(1),model.validate(2)
    i+=1

model.search_for_best_and_worst()

worst=model.get_worst_images()
best=model.get_best_images()

worst_digits_figure = plt.figure()
worst_digits_plots = [worst_digits_figure.add_subplot(4, 5, i+1) for i in xrange(20)]
worst_digits_images = [worst_digits_plots[2*i].imshow(test_images[worst[i]].reshape(28,28), cmap = matplotlib.cm.Greys) for i in xrange(10)]
worst_digits_pies_plots = [worst_digits_plots[2*i+1].pie(model.get_percent(test_images[worst[i]])[0].tolist(),labels=[j for j in xrange(10)]) for i in xrange(10)]
worst_digits_figure.savefig("worst.png")

best_digits_figure = plt.figure()
best_digits_plots = [best_digits_figure.add_subplot(4, 5, i+1) for i in xrange(20)]
best_digits_images = [best_digits_plots[2*i].imshow(test_images[best[i]].reshape(28,28), cmap = matplotlib.cm.Greys) for i in xrange(10)]
best_digits_pies_plots = [best_digits_plots[2*i+1].pie(model.get_percent(test_images[best[i]])[0].tolist(),labels=[j for j in xrange(10)]) for i in xrange(10)]
best_digits_figure.savefig("best.png")
