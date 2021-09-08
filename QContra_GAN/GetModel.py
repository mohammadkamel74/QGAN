# -*- coding: utf-8 -*-


def GetModel(str_model, z_size, img_size, g_conv_dim, d_conv_dim, g_spectral_norm, d_spectral_norm, attention, attention_after_nth_gen_block, attention_after_nth_dis_block, conditional_strategy,
                            num_classes, hypersphere_dim, nonlinear_embed, normalize_embed, needs_init, mixed_precision):
    'Models: DCGAN, QDCGAN, MidQDCGAN'
    print('Model:', str_model)
    print()
    if str_model == 'DCGAN':
        from models.dcgan import Generator, Discriminator
        return (Generator(z_dim=z_size, img_size=img_size, channel=3, g_conv_dim=g_conv_dim, g_spectral_norm=g_spectral_norm, attention=attention,
                            attention_after_nth_gen_block=attention_after_nth_gen_block, conditional_strategy=conditional_strategy,
                            num_classes=num_classes, needs_init=needs_init, mixed_precision=mixed_precision), 
                Discriminator(img_size=img_size, d_conv_dim=d_conv_dim, d_spectral_norm=d_spectral_norm, attention=attention,
                                attention_after_nth_dis_block=attention_after_nth_dis_block, conditional_strategy=conditional_strategy,
                                hypersphere_dim=hypersphere_dim, num_classes=num_classes, nonlinear_embed=nonlinear_embed,
                                normalize_embed=normalize_embed, mixed_precision=mixed_precision, channel=3, needs_init=needs_init))

    elif str_model == 'QDCGAN':
        from models.dcgan import QuaternionGenerator, QuaternionDiscriminator
        return (QuaternionGenerator(z_dim=z_size, img_size=img_size, channel=3, g_conv_dim=g_conv_dim, g_spectral_norm=g_spectral_norm, attention=attention,
                            attention_after_nth_gen_block=attention_after_nth_gen_block, conditional_strategy=conditional_strategy,
                            num_classes=num_classes, needs_init=needs_init, mixed_precision=mixed_precision), 
                QuaternionDiscriminator(img_size=img_size, d_conv_dim=d_conv_dim, d_spectral_norm=d_spectral_norm, attention=attention,
                                attention_after_nth_dis_block=attention_after_nth_dis_block, conditional_strategy=conditional_strategy,
                                hypersphere_dim=hypersphere_dim, num_classes=num_classes, nonlinear_embed=nonlinear_embed,
                                normalize_embed=normalize_embed, mixed_precision=mixed_precision, channel=4, needs_init=needs_init))

    elif str_model == 'ResGAN':
        from models.resnet import Generator, Discriminator
        return (Generator(z_dim=z_size, img_size=img_size, channel=3, g_conv_dim=g_conv_dim, g_spectral_norm=g_spectral_norm, attention=attention,
                            attention_after_nth_gen_block=attention_after_nth_gen_block, conditional_strategy=conditional_strategy,
                            num_classes=num_classes, needs_init=needs_init, mixed_precision=mixed_precision), 
                Discriminator(img_size=img_size, d_conv_dim=d_conv_dim, d_spectral_norm=d_spectral_norm, attention=attention,
                                attention_after_nth_dis_block=attention_after_nth_dis_block, conditional_strategy=conditional_strategy,
                                hypersphere_dim=hypersphere_dim, num_classes=num_classes, nonlinear_embed=nonlinear_embed,
                                normalize_embed=normalize_embed, mixed_precision=mixed_precision, channel=3, needs_init=needs_init))

    elif str_model == 'MidQDCGAN':
        from models.dcgan import MidQuaternionGenerator, MidQuaternionDiscriminator
        return (MidQuaternionGenerator(z_dim=z_size, img_size=img_size, channel=4, g_conv_dim=g_conv_dim, g_spectral_norm=g_spectral_norm, attention=attention,
                            attention_after_nth_gen_block=attention_after_nth_gen_block, conditional_strategy=conditional_strategy,
                            num_classes=num_classes, needs_init=needs_init, mixed_precision=mixed_precision), 
                MidQuaternionDiscriminator(img_size=img_size, d_conv_dim=d_conv_dim, d_spectral_norm=d_spectral_norm, attention=attention,
                                attention_after_nth_dis_block=attention_after_nth_dis_block, conditional_strategy=conditional_strategy,
                                hypersphere_dim=hypersphere_dim, num_classes=num_classes, nonlinear_embed=nonlinear_embed,
                                normalize_embed=normalize_embed, mixed_precision=mixed_precision, channel=4, needs_init=needs_init))

    elif str_model == 'Mid-RG-QD':
        from models.dcgan import Generator, MidQuaternionDiscriminator
        return (Generator(z_dim=z_size, img_size=img_size, channel=4, g_conv_dim=g_conv_dim, g_spectral_norm=g_spectral_norm, attention=attention,
                            attention_after_nth_gen_block=attention_after_nth_gen_block, conditional_strategy=conditional_strategy,
                            num_classes=num_classes, needs_init=needs_init, mixed_precision=mixed_precision), 
                MidQuaternionDiscriminator(img_size=img_size, d_conv_dim=d_conv_dim, d_spectral_norm=d_spectral_norm, attention=attention,
                                attention_after_nth_dis_block=attention_after_nth_dis_block, conditional_strategy=conditional_strategy,
                                hypersphere_dim=hypersphere_dim, num_classes=num_classes, nonlinear_embed=nonlinear_embed,
                                normalize_embed=normalize_embed, mixed_precision=mixed_precision, channel=4, needs_init=needs_init))

    elif str_model == 'MidRes-RG-QD':
        from models.resnet import Generator, MidQuaternionDiscriminator
        return (Generator(z_dim=z_size, img_size=img_size, channel=4, g_conv_dim=g_conv_dim, g_spectral_norm=g_spectral_norm, attention=attention,
                            attention_after_nth_gen_block=attention_after_nth_gen_block, conditional_strategy=conditional_strategy,
                            num_classes=num_classes, needs_init=needs_init, mixed_precision=mixed_precision), 
                MidQuaternionDiscriminator(img_size=img_size, d_conv_dim=d_conv_dim, d_spectral_norm=d_spectral_norm, attention=attention,
                                attention_after_nth_dis_block=attention_after_nth_dis_block, conditional_strategy=conditional_strategy,
                                hypersphere_dim=hypersphere_dim, num_classes=num_classes, nonlinear_embed=nonlinear_embed,
                                normalize_embed=normalize_embed, mixed_precision=mixed_precision, channel=4, needs_init=needs_init))

    

    else:
        raise ValueError ('Model not implemented, check allowed models (-help) \n \
             Models: DCGAN, QDCGAN, ...')
        
        
