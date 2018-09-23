# -*- coding: utf-8 -*-



from keras.layers import Input

from cycleGAN import CycleGAN
import net_utils


class CycleGAN_Unet(CycleGAN):

    def __init__(self, config):
        super().__init__(config)

    def build_model(self, input_shape):
        self.input_A = Input(shape=input_shape, name="input_A")
        self.input_B = Input(shape=input_shape, name="input_B")
        self.input_D = Input(shape=input_shape, name="input_D")

        # model definition of discriminators
        self.Dy = net_utils.discriminator(input_shape, base_name="Dy", use_res=self.config.USE_RES)
        self.Dx = net_utils.discriminator(input_shape, base_name="Dx", use_res=self.config.USE_RES)

        # model definition of generators
        self.G = net_utils.mapping_function_Unet(input_shape, base_name="G",
                                                 num_res_blocks=self.config.NUMBER_RESIDUAL_BLOCKS)
        self.F = net_utils.mapping_function_Unet(input_shape, base_name="F",
                                                num_res_blocks=self.config.NUMBER_RESIDUAL_BLOCKS)