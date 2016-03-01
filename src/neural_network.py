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

import math
import autograd.numpy.numpy_wrapper as anp
from autograd import grad
import numpy
from skimage import transform

init_scale=0.05
batch_size = 32
learning_rate=0.1
dtype = numpy.float64

class NeuralNetworkModel:
    def __init__(self,color,hidden,imgSize,training_set_size,trainImages,trainLabels,test_set_size,testImages,testLabels,rotate):
        self.color=color;
        self.hidden=hidden
        self.imgSize=imgSize
        self.training_set_size=training_set_size
        self.test_set_size=test_set_size
        self.train_images = trainImages
        self.test_images = testImages
        self.train_labels = trainLabels
        self.test_labels = testLabels
        self.best_images=[-1 for i in xrange(10)]
        self.worst_images=[-1 for i in xrange(10)]
        self.best_images_percent=[0 for i in xrange(10)]
        self.worst_images_percent=[1 for i in xrange(10)]
        self.W_vector=[]
        a=imgSize
        for i in hidden:
            self.W_vector.append(init_scale * 2 * (numpy.random.random((a, i)) - 0.5))
            a=i
        self.W_vector.append(init_scale * 2 * (numpy.random.random((a, 10)) - 0.5))

    def scale_and_rotate_image(self,image, angle_range=15.0, scale_range=0.1):
        angle = 2 * angle_range * numpy.random.random() - angle_range
        scale = 1 + 2 * scale_range * numpy.random.random() - scale_range

        tf_rotate = transform.SimilarityTransform(rotation=numpy.deg2rad(angle))
        tf_scale = transform.SimilarityTransform(scale=scale)
        tf_shift = transform.SimilarityTransform(translation=[-14, -14])
        tf_shift_inv = transform.SimilarityTransform(translation=[14, 14])

        if not self.color:
            image = transform.warp(image.reshape([math.sqrt(self.imgSize), math.sqrt(self.imgSize)]),
                               (tf_shift + tf_scale + tf_rotate + tf_shift_inv).inverse)
        else:
            im1=image[0:self.imgSize/3]
            im2=image[self.imgSize/3:(self.imgSize*2)/3]
            im3=image[(self.imgSize*2)/3:]
            im1 = transform.warp(im1.reshape([math.sqrt(self.imgSize/3), math.sqrt(self.imgSize/3)]),
                               (tf_shift + tf_scale + tf_rotate + tf_shift_inv).inverse)
            im2 = transform.warp(im2.reshape([math.sqrt(self.imgSize/3), math.sqrt(self.imgSize/3)]),
                               (tf_shift + tf_scale + tf_rotate + tf_shift_inv).inverse)
            im3 = transform.warp(im3.reshape([math.sqrt(self.imgSize/3), math.sqrt(self.imgSize/3)]),
                               (tf_shift + tf_scale + tf_rotate + tf_shift_inv).inverse)
            image=numpy.concatenate((im1,im2))
            image=numpy.concatenate((image,im3))
        return image.reshape(1,self.imgSize)

    def dropout(self,vec):
        rng = numpy.random.RandomState(1)
        mask = rng.binomial(size=vec.shape, n=1, p=0.9)
        return vec*mask

    def relu(self,vec):
        return anp.log(1+anp.exp(vec))

    def softmax(self,vec):
        e = anp.exp(vec)
        e_sum = anp.sum(e, 1)
        dist = (e.T / e_sum).T
        return dist

    def cost(self,IMG_BATCH, WW, LABELS_BATCH):
        k=0
        mean=0
        sqrtvar=1
        for i in WW:
            if k==0:
                k=1
            else:
                IMG_BATCH=self.relu(IMG_BATCH)
                mean=anp.mean(IMG_BATCH)
                sqrtvar=anp.sqrt(anp.var(IMG_BATCH))
                IMG_BATCH-=mean
                IMG_BATCH/=sqrtvar
            IMG_BATCH=self.dropout(IMG_BATCH)
            IMG_BATCH=anp.dot(IMG_BATCH,i)
            IMG_BATCH*=sqrtvar
            IMG_BATCH+=mean
        softmaxed = self.softmax(IMG_BATCH)
        log_of_softmaxed = anp.log(softmaxed)
        errors = log_of_softmaxed * LABELS_BATCH
        global_cost = -anp.sum(errors) / batch_size
        return global_cost

    def train(self):
        g1 = grad(self.cost, 1)
        permutation = numpy.random.permutation(self.training_set_size)
        i = 1
        IMG_BATCH = 0
        LABELS_BATCH = 0
        for k in permutation:
            if i == 1:
                IMG_BATCH = self.scale_and_rotate_image(self.train_images[k])
                x = int(self.train_labels[k])
                LABELS_BATCH = anp.array([],dtype=dtype)
                for j in range(0, 10):
                    if j == x:
                        LABELS_BATCH = anp.append(LABELS_BATCH, numpy.array(1,dtype=dtype))
                    else:
                        LABELS_BATCH = anp.append(LABELS_BATCH, numpy.array(0,dtype=dtype))
            else:
                IMG_BATCH = anp.concatenate((IMG_BATCH, self.scale_and_rotate_image(self.train_images[k])))
                x = int(self.train_labels[k])
                for j in range(0, 10):
                    if j == x:
                        LABELS_BATCH = anp.append(LABELS_BATCH, anp.array(1,dtype=dtype))
                    else:
                        LABELS_BATCH = anp.append(LABELS_BATCH, anp.array(0,dtype=dtype))
            if i == batch_size:
                i = 0
                delta = g1(IMG_BATCH.reshape(batch_size, self.imgSize), self.W_vector, LABELS_BATCH.reshape(batch_size, 10))
                for p in range(0,len(delta)):
                    self.W_vector[p] -= (delta[p] * learning_rate)

            i += 1

    def validate(self,mode):
        if mode==1:
            test_images=self.train_images[:5000]
            test_labels=self.train_labels[:5000]
        if mode==2:
            test_images=self.test_images
            test_labels=self.test_labels
        good = 0
        bad = 0
        temp = 0
        maxi = 0
        max_num = 0
        while temp < len(test_labels):
            k=0
            IMG=test_images[temp].reshape(1, self.imgSize)
            mean=0
            sqrtvar=1
            for i in self.W_vector:
                if k==0:
                    k=1
                else:
                    IMG=self.relu(IMG)
                    mean=numpy.mean(IMG)
                    sqrtvar=numpy.sqrt(numpy.var(IMG))
                    IMG-=mean
                    IMG/=sqrtvar
                IMG=anp.dot(IMG,i)
                IMG*=sqrtvar
                IMG+=mean
            IMG=self.softmax(IMG)
            for i in range(0, 10):
                if IMG[0][i] > maxi:
                    max_num = i
                    maxi = IMG[0][i]
            if max_num == int(test_labels[temp]):
                good += 1
            else:
                bad += 1
            max_num=0
            maxi=0
            temp += 1
        if mode==1:
            test_images=self.train_images[self.test_set_size-5000:]
            test_labels=self.train_labels[self.test_set_size-5000:]
            temp = 0
            maxi = 0
            max_num = 0
            while temp < len(test_labels):
                k=0
                IMG=test_images[temp].reshape(1, self.imgSize)
                mean=0
                sqrtvar=1
                for i in self.W_vector:
                    if k==0:
                        k=1
                    else:
                        IMG=self.relu(IMG)
                        mean=numpy.mean(IMG)
                        sqrtvar=numpy.sqrt(numpy.var(IMG))
                        IMG-=mean
                        IMG/=sqrtvar
                    IMG=anp.dot(IMG,i)
                    IMG*=sqrtvar
                    IMG+=mean
                IMG=self.softmax(IMG)
                for i in range(0, 10):
                    if IMG[0][i] > maxi:
                        max_num = i
                        maxi = IMG[0][i]
                if max_num == int(test_labels[temp]):
                    good += 1
                else:
                    bad += 1
                max_num=0
                maxi=0
                temp += 1
        return good * 1.0 / (good + bad) * 1.0

    def search_for_best_and_worst(self):
        test_images=self.test_images
        test_labels=self.test_labels
        good = 0
        bad = 0
        temp = 0
        maxi = 0
        max_num = 0
        while temp < len(test_labels):
            k=0
            IMG=test_images[temp].reshape(1, self.imgSize)
            mean=0
            sqrtvar=1
            for i in self.W_vector:
                if k==0:
                    k=1
                else:
                    IMG=self.relu(IMG)
                    mean=numpy.mean(IMG)
                    sqrtvar=numpy.sqrt(numpy.var(IMG))
                    IMG-=mean
                    IMG/=sqrtvar
                IMG=anp.dot(IMG,i)
                IMG*=sqrtvar
                IMG+=mean
            IMG=self.softmax(IMG)
            for i in range(0, 10):
                if IMG[0][i] > maxi:
                    max_num = i
                    maxi = IMG[0][i]
            if max_num == int(test_labels[temp]):
                good += 1
            else:
                bad += 1
            if IMG[0][int(test_labels[temp])]<self.worst_images_percent[int(test_labels[temp])]:
                    self.worst_images[int(test_labels[temp])]=temp;
                    self.worst_images_percent[int(test_labels[temp])]=IMG[0][int(test_labels[temp])]
            elif IMG[0][int(test_labels[temp])]>self.best_images_percent[int(test_labels[temp])]:
                self.best_images[int(test_labels[temp])]=temp;
                self.best_images_percent[int(test_labels[temp])]=IMG[0][int(test_labels[temp])]
            max_num=0
            maxi=0
            temp += 1

    def get_best_images(self):
        return self.best_images

    def get_worst_images(self):
        return self.worst_images

    def get_percent(self,img):
        IMG=img.reshape(1, self.imgSize)
        k=0
        mean=0
        sqrtvar=1
        for i in self.W_vector:
            if k==0:
                k=1
            else:
                IMG=self.relu(IMG)
                mean=numpy.mean(IMG)
                sqrtvar=numpy.sqrt(numpy.var(IMG))
                IMG-=mean
                IMG/=sqrtvar
            IMG=anp.dot(IMG,i)
            IMG*=sqrtvar
            IMG+=mean
        IMG=self.softmax(IMG)
        return IMG


