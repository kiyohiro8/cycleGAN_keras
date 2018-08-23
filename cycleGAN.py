# -*- coding: utf-8 -*-


import time
import datetime
import os

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input
from keras import optimizers

import utils
import net_utils

class CycleGAN():
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        input_shape = self.config.INPUT_SHAPE

        self.input_A = Input(shape=input_shape, name="input_A")
        self.input_B = Input(shape=input_shape, name="input_B")
        self.input_D = Input(shape=input_shape, name="input_D")

        # model definition of discriminators
        self.Dy = net_utils.discriminator(input_shape, base_name="Dy")
        self.Dx = net_utils.discriminator(input_shape, base_name="Dx")

        # model definition of generators
        self.G = net_utils.mapping_function(input_shape, base_name="G")
        self.F = net_utils.mapping_function(input_shape, base_name="F")

    def train(self):
        # construct training model of discriminators
        Dy_out = self.Dy(self.input_D)
        Dx_out = self.Dx(self.input_D)

        # construct training model of generator G (A → B)
        A2B = self.G(self.input_A)
        Dy_for_G = self.Dy(A2B)
        A2B2A = self.F(A2B)

        # construct training model of generator F (B → A)
        B2A = self.F(self.input_B)
        Dx_for_F = self.Dx(B2A)
        B2A2B = self.G(B2A)

        # compile discriminators
        print("Compile Discriminators")
        optim = optimizers.Adam(lr=self.config.LEARNING_RATE, beta_1=self.config.BETA_1)
        model_Dx = Model(inputs=[self.input_D], outputs=[Dx_out])
        model_Dx.compile(optimizer=optim,
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        model_Dy = Model(inputs=[self.input_D], outputs=[Dy_out])
        model_Dy.compile(optimizer=optim,
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

        # compile generator
        print("Compile Generator")
        generator = Model(inputs=[self.input_A, self.input_B], outputs=[Dy_for_G, Dx_for_F, A2B2A, B2A2B])
        generator = utils.set_trainable(generator, prefix_list=["Dy", "Dx"], trainable=False)
        generator.compile(optimizer=optim,
                          loss={"Dy": "binary_crossentropy", "Dx": "binary_crossentropy",
                                "F": "mae", "G": "mae"},
                          loss_weights={"Dy": 1, "Dx": 1,
                                        "F": self.config.LAMBDA, "G": self.config.LAMBDA})

        # construct one-way mapping model for inference
        inference_model_G = Model(inputs=[self.input_A], outputs=[A2B])
        inference_model_F = Model(inputs=[self.input_B], outputs=[B2A])

        now = datetime.datetime.now()
        datetime_sequence = "{0}{1:02d}{2:02d}_{3:02}{4:02d}".format(str(now.year)[-2:], now.month, now.day ,
                                                                    now.hour, now.minute)

        dataset_name_A = self.config.DATASET_A
        dataset_name_B = self.config.DATASET_B

        datasetA = utils.data_generator(os.path.join(self.config.DATA_DIR, dataset_name_A, self.config.DATA_EXT), self.config.BATCH_SIZE)
        datasetB = utils.data_generator(os.path.join(self.config.DATA_DIR, dataset_name_B, self.config.DATA_EXT), self.config.BATCH_SIZE)

        output_name_1 = dataset_name_A + "2" + dataset_name_B
        output_name_2 = dataset_name_B + "2" + dataset_name_A

        experiment_dir = os.path.join(self.config.RESULT_DIR, datetime_sequence)

        sample_output_dir_1 = os.path.join(experiment_dir, "sample", output_name_1)
        sample_output_dir_2 = os.path.join(experiment_dir, "sample", output_name_2)
        weights_output_dir_1 = os.path.join(experiment_dir, "weights", output_name_1)
        weights_output_dir_2 = os.path.join(experiment_dir, "weights", output_name_2)

        os.makedirs(sample_output_dir_1, exist_ok=True)
        os.makedirs(sample_output_dir_2, exist_ok=True)
        os.makedirs(weights_output_dir_1, exist_ok=True)
        os.makedirs(weights_output_dir_2, exist_ok=True)

        start_time = time.time()
        met_curve = pd.DataFrame(columns=["counter", "loss_Dx", "loss_Dy", "loss_G"])
        counter = 0

        for epoch in range(self.config.EPOCH):
            for iter in range(self.config.ITER_PER_EPOCH):
                # generate minibatch
                batch_files_A = next(datasetA)
                batch_A = np.array([utils.get_image(file, input_hw=self.config.INPUT_SHAPE[0]) for file in batch_files_A])
                batch_files_B = next(datasetB)
                batch_B = np.array([utils.get_image(file, input_hw=self.config.INPUT_SHAPE[0]) for file in batch_files_B])
                # update generator's gradients
                loss_g = generator.train_on_batch(x={"input_A": batch_A, "input_B": batch_B},
                                                  y={"Dy": np.zeros(self.config.BATCH_SIZE),
                                                     "Dx": np.zeros(self.config.BATCH_SIZE),
                                                     "G": batch_B, "F": batch_A})

                # update Dx's gradients
                X = inference_model_F.predict(batch_B)
                X = np.append(batch_A, X, axis=0)
                y = [1] * len(batch_B) + [0] * len(batch_A)
                y = np.array(y)
                loss_d_x, acc_d_x = model_Dx.train_on_batch(X, y)

                # update Dy's gradients
                X = inference_model_G.predict(batch_A)
                X = np.append(batch_B, X, axis=0)
                y = [1] * len(batch_A) + [0] * len(batch_B)
                y = np.array(y)
                loss_d_y, acc_d_y = model_Dy.train_on_batch(X, y)

                elapsed = time.time() - start_time

                print("epoch {0} {1}/{2} loss_d_x:{3:.4f} loss_d_y:{4:.4f} "
                      "loss_g:{5:.4f} {7:.2f}秒".format(epoch, iter, 1000, loss_d_x, loss_d_y,
                                                        loss_g[0], loss_g[4], elapsed))

                if counter % 10 == 0:
                    temp_df = pd.DataFrame(np.array([counter, loss_d_x, loss_d_y, loss_g[0]]))
                    met_curve = pd.concat([met_curve, temp_df], axis=0)

                if counter % 500 == 0:
                    sample_1 = inference_model_G.predict(batch_A)
                    sample_2 = inference_model_F.predict(batch_B)
                    combine_sample_1 = np.concatenate([batch_A[0], sample_1[0]], axis=1)
                    combine_sample_2 = np.concatenate([batch_B[0], sample_2[0]], axis=1)

                    file_1 = "{0}_{1}.jpg".format(epoch, counter)
                    utils.output_sample_image(os.path.join(sample_output_dir_1, file_1), combine_sample_1)
                    file_2 = "{0}_{1}.jpg".format(epoch, counter)
                    utils.output_sample_image(os.path.join(sample_output_dir_2, file_2), combine_sample_2)

                if counter % 10000 == 0:

                    net_utils.save_weights(inference_model_G, weights_output_dir_1, counter)
                    net_utils.save_weights(inference_model_F, weights_output_dir_2, counter)
                    met_curve.to_csv(os.path.join(experiment_dir,
                                                  dataset_name_A + "_" + dataset_name_B + ".csv"),
                                     index=False)

                counter += 1

            met_curve.to_csv(os.path.join(experiment_dir,
                                          dataset_name_A + "_" + dataset_name_B + ".csv"),
                             index=False)

    def inference(self, target_dir):
        #TODO Imprement
        pass









