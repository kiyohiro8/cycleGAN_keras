# -*- coding: utf-8 -*-


import time
import datetime
import os
from glob import glob

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
        #self.build_model()

    def build_model(self, input_shape):
        self.input_A = Input(shape=input_shape, name="input_A")
        self.input_B = Input(shape=input_shape, name="input_B")
        self.input_D = Input(shape=input_shape, name="input_D")

        # model definition of discriminators
        self.Dy = net_utils.discriminator(input_shape, base_name="Dy", use_res=self.config.USE_RES)
        self.Dx = net_utils.discriminator(input_shape, base_name="Dx", use_res=self.config.USE_RES)

        # model definition of generators
        self.G = net_utils.mapping_function(input_shape, base_name="G",
                                            num_res_blocks=self.config.NUMBER_RESIDUAL_BLOCKS)
        self.F = net_utils.mapping_function(input_shape, base_name="F",
                                            num_res_blocks=self.config.NUMBER_RESIDUAL_BLOCKS)

    def train(self):
        """
        学習を行うメソッドです。
        :return: None
        """
        self.build_model(input_shape=self.config.INPUT_SHAPE)
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
        self.Dy.trainable = False
        self.Dx.trainable = False
        self.Dy = net_utils.set_trainable(self.Dy, ["Dy"], trainable=False)
        self.Dx = net_utils.set_trainable(self.Dx, ["Dx"], trainable=False)

        # compile generator
        print("Compile Generator")
        generator = Model(inputs=[self.input_A, self.input_B], outputs=[Dy_for_G, Dx_for_F, A2B2A, B2A2B])
        generator.compile(optimizer=optim,
                          loss={"Dy": "binary_crossentropy", "Dx": "binary_crossentropy",
                                "F": "mae", "G": "mae"},
                          loss_weights={"Dy": 1, "Dx": 1,
                                        "F": self.config.LAMBDA, "G": self.config.LAMBDA})

        # construct one-way mapping model for inference
        inference_model_G = Model(inputs=[self.input_A], outputs=[A2B])
        inference_model_F = Model(inputs=[self.input_B], outputs=[B2A])

        self.training_iteration(Dx=model_Dx, Dy=model_Dy, generator=generator,
                                G=inference_model_G, F=inference_model_F)


    def inference(self, model_path, target_dir):
        """

        :param model_path:
        :param target_dir:
        :return:
        """
        # TODO: ファイルごとにモデルを作り直しているため非常に遅い。なんとかする。
        path_list = glob(os.path.join(target_dir, self.config.DATA_EXT))

        output_dir_name = os.path.join("translated", os.path.basename(target_dir))
        os.makedirs(output_dir_name, exist_ok=True)

        for image_path in path_list:
            image = utils.imread(image_path)
            shape = image.shape
            input_layer = Input(shape=shape)
            G = net_utils.mapping_function(shape, base_name="G",
                                           num_res_blocks=self.config.NUMBER_RESIDUAL_BLOCKS)
            A2B = G(input_layer)
            inference_model = Model(inputs=[input_layer], outputs=[A2B])
            inference_model.load_weights(model_path, by_name=True)

            image = np.array([image])
            translated_image =inference_model.predict(image)
            name = os.path.basename(image_path)
            output_path = os.path.join(output_dir_name, name)
            utils.output_sample_image(output_path, translated_image[0])

    def resume_train(self):
        Dx_filepath = os.path.join(self.config.RESULT_DIR, self.config.RESUME_FROM,
                                   "weights/resume", "Dx"+str(self.config.COUNTER) + ".hdf5")
        Dy_filepath = os.path.join(self.config.RESULT_DIR, self.config.RESUME_FROM,
                                   "weights/resume", "Dy"+str(self.config.COUNTER) + ".hdf5")
        generator_filepath = os.path.join(self.config.RESULT_DIR, self.config.RESUME_FROM,
                                   "weights/resume", "generator"+str(self.config.COUNTER) + ".hdf5")

        self.build_model(self.config.INPUT_SHAPE)
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

        optim = optimizers.Adam(lr=self.config.LEARNING_RATE, beta_1=self.config.BETA_1)
        model_Dx = Model(inputs=[self.input_D], outputs=[Dx_out])
        model_Dx.load_weights(Dx_filepath, by_name=True)
        model_Dy = Model(inputs=[self.input_D], outputs=[Dy_out])
        model_Dy.load_weights(Dy_filepath, by_name=True)

        print("Compile Discriminators")
        model_Dx.compile(optimizer=optim,
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        model_Dy.compile(optimizer=optim,
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        self.Dy.trainable = False
        self.Dx.trainable = False
        self.Dy = net_utils.set_trainable(self.Dy, ["Dy"], trainable=False)
        self.Dx = net_utils.set_trainable(self.Dx, ["Dx"], trainable=False)

        # compile generator
        print("Compile Generator")
        generator = Model(inputs=[self.input_A, self.input_B], outputs=[Dy_for_G, Dx_for_F, A2B2A, B2A2B])
        generator.load_weights(generator_filepath)
        generator.compile(optimizer=optim,
                          loss={"Dy": "binary_crossentropy", "Dx": "binary_crossentropy",
                                "F": "mae", "G": "mae"},
                          loss_weights={"Dy": 1, "Dx": 1,
                                        "F": self.config.LAMBDA, "G": self.config.LAMBDA})

        # construct one-way mapping model for inference
        inference_model_G = Model(inputs=[self.input_A], outputs=[A2B])
        inference_model_F = Model(inputs=[self.input_B], outputs=[B2A])

        self.training_iteration(Dx=model_Dx, Dy=model_Dy, generator=generator,
                                G=inference_model_G, F=inference_model_F,
                                counter=self.config.COUNTER)


    def training_iteration(self, Dx, Dy, generator, G, F, counter=0):

        now = datetime.datetime.now()
        datetime_sequence = "{0}{1:02d}{2:02d}_{3:02}{4:02d}".format(str(now.year)[-2:], now.month, now.day ,
                                                                    now.hour, now.minute)

        datasetA = utils.data_generator(os.path.join(self.config.DATA_DIR, self.config.DATASET_A,
                                                     self.config.DATA_EXT), self.config.BATCH_SIZE)
        datasetB = utils.data_generator(os.path.join(self.config.DATA_DIR, self.config.DATASET_B,
                                                     self.config.DATA_EXT), self.config.BATCH_SIZE)

        output_name_1 = self.config.DATASET_A + "2" + self.config.DATASET_B
        output_name_2 = self.config.DATASET_B + "2" + self.config.DATASET_A

        experiment_dir = os.path.join(self.config.RESULT_DIR, datetime_sequence)

        sample_output_dir_1 = os.path.join(experiment_dir, "sample", output_name_1)
        sample_output_dir_2 = os.path.join(experiment_dir, "sample", output_name_2)
        weights_output_dir_1 = os.path.join(experiment_dir, "weights", output_name_1)
        weights_output_dir_2 = os.path.join(experiment_dir, "weights", output_name_2)
        weights_output_dir_resume = os.path.join(experiment_dir, "weights", "resume")

        os.makedirs(sample_output_dir_1, exist_ok=True)
        os.makedirs(sample_output_dir_2, exist_ok=True)
        os.makedirs(weights_output_dir_1, exist_ok=True)
        os.makedirs(weights_output_dir_2, exist_ok=True)
        os.makedirs(weights_output_dir_resume, exist_ok=True)

        start_time = time.time()
        met_curve = pd.DataFrame(columns=["counter", "loss_Dx", "loss_Dy", "loss_G",
                                          "adversarial_loss", "cycle_loss"])

        for epoch in range(self.config.EPOCH):
            for iter in range(self.config.ITER_PER_EPOCH):
                # generate minibatch
                batch_files_A = next(datasetA)
                batch_A = np.array([utils.get_image(file, input_hw=self.config.INPUT_SHAPE[0]) for file in batch_files_A])
                batch_files_B = next(datasetB)
                batch_B = np.array([utils.get_image(file, input_hw=self.config.INPUT_SHAPE[0]) for file in batch_files_B])
                # update generator's gradients
                loss_g = generator.train_on_batch(x={"input_A": batch_A, "input_B": batch_B},
                                                  y={"Dy": np.ones(self.config.BATCH_SIZE),
                                                     "Dx": np.ones(self.config.BATCH_SIZE),
                                                     "G": batch_B, "F": batch_A})

                # update Dx's gradients
                X = F.predict(batch_B)
                X = np.append(batch_A, X, axis=0)
                y = [0] * len(batch_B) + [1] * len(batch_A)
                y = np.array(y)
                loss_d_x, acc_d_x = Dx.train_on_batch(X, y)

                # update Dy's gradients
                X = G.predict(batch_A)
                X = np.append(batch_B, X, axis=0)
                y = [0] * len(batch_A) + [1] * len(batch_B)
                y = np.array(y)
                loss_d_y, acc_d_y = Dy.train_on_batch(X, y)

                elapsed = time.time() - start_time

                print("epoch {0} {1}/{2} loss_d_x:{3:.4f} loss_d_y:{4:.4f} "
                      "loss_g:{5:.4f} {7:.2f}秒".format(epoch, iter, 1000, loss_d_x, loss_d_y,
                                                        loss_g[0], loss_g[4], elapsed))

                if counter % 10 == 0:
                    temp_df = pd.DataFrame({"counter":[counter], "loss_Dx":[loss_d_x],
                                            "loss_Dy":[loss_d_y], "loss_G":[loss_g[0]],
                                            "adversarial_loss":[loss_g[1] + loss_g[2]],
                                            "cycle_loss":[loss_g[3] + loss_g[4]]})
                    met_curve = pd.concat([met_curve, temp_df], axis=0)

                if counter % 200 == 0:
                    sample_1 = G.predict(batch_A)
                    sample_2 = F.predict(batch_B)
                    combine_sample_1 = np.concatenate([batch_A[0], sample_1[0]], axis=1)
                    combine_sample_2 = np.concatenate([batch_B[0], sample_2[0]], axis=1)

                    file_1 = "{0}_{1}.jpg".format(epoch, counter)
                    utils.output_sample_image(os.path.join(sample_output_dir_1, file_1), combine_sample_1)
                    file_2 = "{0}_{1}.jpg".format(epoch, counter)
                    utils.output_sample_image(os.path.join(sample_output_dir_2, file_2), combine_sample_2)

                if counter % 1000 == 0:

                    net_utils.save_weights(G, weights_output_dir_1, counter)
                    net_utils.save_weights(F, weights_output_dir_2, counter)

                    net_utils.save_weights(generator, weights_output_dir_resume, counter, base_name="generator")
                    net_utils.save_weights(Dx, weights_output_dir_resume, counter, base_name="Dy")
                    net_utils.save_weights(Dy, weights_output_dir_resume, counter, base_name="Dx")

                    met_curve.to_csv(os.path.join(experiment_dir,
                                                  self.config.DATASET_A + "_"
                                                  + self.config.DATASET_B + ".csv"),
                                     index=False)

                counter += 1


        net_utils.save_weights(G, weights_output_dir_1, counter)
        net_utils.save_weights(F, weights_output_dir_2, counter)

        net_utils.save_weights(generator, weights_output_dir_resume, counter, base_name="generator")
        net_utils.save_weights(Dx, weights_output_dir_resume, counter, base_name="Dy")
        net_utils.save_weights(Dy, weights_output_dir_resume, counter, base_name="Dx")

        met_curve.to_csv(os.path.join(experiment_dir,
                                      self.config.DATASET_A + "_"
                                      + self.config.DATASET_B + ".csv"),
                         index=False)












