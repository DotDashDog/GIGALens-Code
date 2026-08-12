halo_model = Prior(
    #mass.epl.EPL(),
    # mass.nfw.NFW_ELLIPSE_EINSTEIN(),
    mass.nfw_ellipse_slope.NFW_ELLIPSE_SLOPE(),
    # mass.bpl.BPL(),
    # mass.piemd.PIEMD(),
    # best_model['lens_mass']['0'] |
    dict(
        center_x = tfd.Normal(6.69965551, 1),
        center_y = tfd.Normal(4.80431651, 1),
        e1 = tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        # Rs = tfd.Uniform(20, 100),
        s_E = tfd.Uniform(0., 0.8),
        theta_E = tfd.Uniform(12, 14),

        # # #gamma = tfd.TruncatedNormal(1.6, 0.5, 1 , 3),
        # gamma = tfd.Uniform(1,3)
        # b = tfd.Uniform(10,30), #tfd.TruncatedNormal(13, 2, 10, 20), #tfd.Normal(13, 1),
        # alpha = tfd.Uniform(1,3),
        # alpha_c = lambda alpha: tfd.Uniform(0, alpha),
        # r_core = lambda theta_E: tfd.TruncatedNormal(theta_E/2, theta_E/4, 0, theta_E)
    )
)


# fixed position, PIEMD
ld_free_ellip_model = Prior(
    mass.piemd.DPIE(),
    # mass.epl.EPL(),
    # best_model['lens_mass']['1'],
    dict(
        center_x = tfd.Normal(11.80977389, 0.1),
        center_y = tfd.Normal(23.0283886, 0.1),
        # theta_E = 1.6730331,
        # r_cut = 2.749177,
        # r_core = 0.2986106,
        # e1 = 0.41781854,
        # e2 = 0.07367268,
        theta_E = tfd.TruncatedNormal(1.6730331, 0.5, 1, 2.5),
        r_cut = tfd.LogNormal(jnp.log(10), 1),
        r_core = tfd.LogNormal(jnp.log(0.05), 0.01),
        # gamma = tfd.TruncatedNormal(2., 0.5, 1, 3),
        e1 = tfd.TruncatedNormal(0, 0.1, -0.5, 0.5),
        e2 = tfd.TruncatedNormal(0, 0.1, -0.5, 0.5), #lambda e1: tfd.TruncatedNormal(0, 0.05, -np.sqrt(0.09-e1*2), np.sqrt(0.09-e1*2)),
    ) #| best_model['lens_mass']['1']
)

shear_model = Prior(mass.shear.Shear(),
                    # best_model['lens_mass']['3'] |
                    dict(
                        gamma1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                        gamma2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                    ))

le_free_model = Prior(#mass.epl.EPL(),
                      mass.piemd.DPIE(),
                      # best_model['lens_mass']['1'] |
                   dict(
                       center_x = tfd.Normal(-21.17580938, 0.1),
                       center_y = tfd.Normal(-24.25810504, 0.1),
                       # e1 = 0.07383415,
                       # e2 = 0.03570823,
                       e1 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                       e2 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                       theta_E = tfd.TruncatedNormal(1.6711541, 0.1, 1, 2.5),
                       # gamma = tfd.TruncatedNormal(2.007689, 0.5, 1, 3)
                       r_cut = tfd.LogNormal(jnp.log(10), 1),
                       r_core = tfd.LogNormal(jnp.log(0.05), 0.01),
                   ) #| best_model['lens_mass']['3']
                   )

group_halo_free_model = Prior(#mass.epl.EPL(),
                              mass.piemd.DPIE(),
                              # best_model['lens_mass']['2'] |
                   dict(
                       center_x = tfd.Normal(-15.10088063, .1),
                       center_y = tfd.Normal(-4.66657821, .1),
                       e1 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                       e2 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                       theta_E = tfd.TruncatedNormal(0.8151327, 0.1, 0.2, 1.5),
                       r_cut = tfd.LogNormal(jnp.log(10), 1),
                       r_core = tfd.LogNormal(jnp.log(0.05), 0.01),
                       # gamma = tfd.TruncatedNormal(2.2266, 0.5, 1, 3)
                   ) #| best_model['lens_mass']['4']
)

upper_right_halo = Prior(mass.epl.EPL(),
                   dict(
                       center_x = tfd.Normal(32, .5),
                       center_y = tfd.Normal(22, .5),
                       e1 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                       e2 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                       theta_E = tfd.LogNormal(jnp.log(1),0.05),
                       gamma = tfd.TruncatedNormal(2, 0.5, 1, 3)
                   )
)

source1_prior = Prior(
    light.combined_profile.CombinedProfile(
        profiles=[
            light.sersic.DoubleSersic(use_lstsq=True, cosmo_sample=False),
            light.sersic.SersicEllipse(use_lstsq=True, cosmo_sample=False)
        ],
        shared_params=[],
        use_lstsq=True,
        cosmo_sample=False
    ),
    dict(
        deflection_ratio = tfd.Uniform(0.5,1),
        # z_source = 0.962,
        center_x_0 = tfd.Normal(7.67187389, 5),
        center_y_0 = tfd.Normal(3.31911655, 5),
        e1_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic_0_0 = tfd.Uniform(.25,10),
        n_sersic_1_0 = tfd.Uniform(.25,10),
        R_sersic_0_0 = tfd.LogNormal(jnp.log(0.4), 0.15),
        R_sersic_1_0 = tfd.LogNormal(jnp.log(0.4), 0.15),
        # beta_0 = tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie = tfd.LogNormal(jnp.log(40), 1),

        center_x_1 = tfd.Normal(10, 5),
        center_y_1 = tfd.Normal(3, 5),
        e1_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic_1 = tfd.Uniform(1,10),
        R_sersic_1 = tfd.LogNormal(jnp.log(0.4), 0.15),
    )
)

source3_prior = Prior(
    light.combined_profile.CombinedProfile(
        profiles=[
            light.sersic.SersicEllipse(use_lstsq=True, cosmo_sample=False),
            light.sersic.SersicEllipse(use_lstsq=True, cosmo_sample=False)
        ],
        shared_params=['center_x', 'center_y', 'e1', 'e2'],
        # shared_params = [],
        use_lstsq=True,
        cosmo_sample=False
    ),
    dict(
        deflection_ratio = tfd.Uniform(0.5,1),
        # z_source = 1.166,
        center_x = tfd.Normal(6.79821086, 5),
        center_y = tfd.Normal(7.91570776, 5),
        # center_x_1 = tfd.Normal(6.79821086, 5),
        # center_y_1 = tfd.Normal(7.91570776, 5),
        e1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        # e1_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        # e2_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic_0 = tfd.Uniform(.25,10),
        R_sersic_0 = tfd.LogNormal(jnp.log(0.4), 0.15),
        n_sersic_1 = tfd.Uniform(.25,10),
        # beta_0 = lambda R_sersic_0: tfd.Normal(R_sersic_0, 0.01),#tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        R_sersic_1 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.05), 0.05),
        # Ie = tfd.LogNormal(jnp.log(40), 1),
    )
)
source45_prior = Prior(
    light.combined_profile.CombinedProfile(
        profiles=[
            light.sersic.SersicEllipse(use_lstsq=True, cosmo_sample=False),
            light.sersic.SersicEllipse(use_lstsq=True, cosmo_sample=False),
        ],
        shared_params=[],
        use_lstsq=True,
        cosmo_sample=False
    ),
    # best_model['source_light']['0'] |
    dict(
        deflection_ratio = 1,
        # z_source = 1.432,
        center_x_0 = tfd.Normal(4.63, 5),
        center_y_0 = tfd.Normal(3.79, 5),
        center_x_1 = tfd.Normal(4.78792923, 5),
        center_y_1 = tfd.Normal(0.84347007, 5),
        e1_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),

        e1_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic_0 = tfd.Uniform(.25,10),
        n_sersic_1 = tfd.Uniform(.25,10),
        R_sersic_0 = tfd.Uniform(1e-3,1),
        R_sersic_1 = tfd.Uniform(1e-3,1),
        # beta_0 = tfd.Uniform(1e-3,1),
        # beta_1 = tfd.Uniform(1e-3,1),
        # Ie_0 = tfd.LogNormal(jnp.log(40), 1),
        # Ie_1 = tfd.LogNormal(jnp.log(40), 1),
    )
)
# source 9
source9_prior = Prior(
    light.sersic.SersicEllipse(use_lstsq=True, cosmo_sample=False),
    dict(
        deflection_ratio = tfd.Uniform(0.75, 1.25),
        # z_source = 1.506,
        center_x = tfd.Normal(-7, 5),
        center_y = tfd.Normal(-13, 5),
        e1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic = tfd.Uniform(.25,10),
        R_sersic = tfd.Uniform(1e-3,1),

        # beta = tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie = tfd.LogNormal(jnp.log(40), 1),
    )
)
source7_prior = Prior(
    light.sersic.SersicEllipse(use_lstsq=True, cosmo_sample=False),
    dict(
        deflection_ratio = tfd.Uniform(1.,1.5),
        # z_source = 1.627,
        center_x = tfd.Normal(0, 5),
        center_y = tfd.Normal(0, 5),
        e1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic = tfd.Uniform(.25,10),
        R_sersic = tfd.Uniform(1e-3,1),
        # Ie = tfd.LogNormal(jnp.log(40), 1),
    )
)
source6_prior = Prior(
    light.sersic.SersicEllipse(use_lstsq=True, cosmo_sample=False),
    dict(
        deflection_ratio = tfd.Uniform(1.,1.5),
        # z_source = 1.656,
        center_x = tfd.Normal(0, 5),
        center_y = tfd.Normal(0, 5),
        e1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic = tfd.Uniform(.25,10),
        R_sersic = tfd.Uniform(1e-3,1),
        # Ie = tfd.LogNormal(jnp.log(40), 1),
    )
)
source1213_prior = Prior(
    light.combined_profile.CombinedProfile(
        profiles=[
                  light.sersic.SersicEllipse(use_lstsq=True, cosmo_sample=False),
                  light.sersic.SersicEllipse(use_lstsq=True, cosmo_sample=False),],
        shared_params=[],
        use_lstsq=True,
        cosmo_sample=False
    ),
    dict(
        # deflection_ratio = 1.2674489,
        deflection_ratio = tfd.Uniform(1,1.5),
        # z_source = 3.086,
        center_x_0 = tfd.Normal(2.906817674636841, 5),
        center_y_0 = tfd.Normal(4.523301601409912, 5),
        center_x_1 = tfd.Normal(6.670745849609375, 5),
        center_y_1 = tfd.Normal(5.6769537925720215, 5),
        e1_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e1_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic_0 = tfd.Uniform(.25,10),
        n_sersic_1 = tfd.Uniform(.25,10),
        R_sersic_0 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        R_sersic_1 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        # beta_0 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie_0 = tfd.LogNormal(jnp.log(40), 1),
        # Ie_1 = tfd.LogNormal(jnp.log(40), 1),
    )
)
source8_prior = Prior(
    light.sersic.SersicEllipse(use_lstsq=True, cosmo_sample=False),
    dict(
        deflection_ratio = tfd.Uniform(1,1.5),
        # z_source = 3.549,
        center_x = tfd.Normal(6.1898108, 5),
        center_y = tfd.Normal(6.8792906, 5),
        e1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic = tfd.Uniform(.25,10),
        R_sersic = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        # beta = tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie = tfd.LogNormal(jnp.log(40), 1),
    )
)
source11_prior = Prior(
    light.sersic.SersicEllipse(use_lstsq=True, cosmo_sample=False),
    dict(
        deflection_ratio = tfd.Uniform(1.,1.5),
        # z_source = 4.090,
        center_x = tfd.Normal(4.4566746, 5),
        center_y = tfd.Normal(2.0770147, 5),
        e1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic = tfd.Uniform(.25,10),
        R_sersic = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie = tfd.LogNormal(jnp.log(40), 1),
    )
)

)