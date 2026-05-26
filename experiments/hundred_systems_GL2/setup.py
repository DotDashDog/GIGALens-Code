def make_default_prior():
    """
    Make the default prior from the original GIGALens paper.
    """
    lens_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
                    gamma=tfd.TruncatedNormal(2, 0.5, 1, 3),
                    e1=tfd.Normal(0, 0.2),
                    e2=tfd.Normal(0, 0.2),
                    center_x=tfd.Normal(0, 0.06),
                    center_y=tfd.Normal(0, 0.06),
                )
            ),
            tfd.JointDistributionNamed(
                dict(gamma1=tfd.Normal(0, 0.1), gamma2=tfd.Normal(0, 0.1))
            ),
        ]
    )
    lens_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
                    n_sersic=tfd.Uniform(0.5, 8),
                    e1=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
                    e2=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
                    center_x=tfd.Normal(0, 0.02),
                    center_y=tfd.Normal(0, 0.02),
                    Ie=tfd.LogNormal(jnp.log(300.0), 0.5),
                )
            )
        ]
    )

    source_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
                    n_sersic=tfd.Uniform(0.5, 8),
                    e1=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
                    e2=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
                    center_x=tfd.Normal(0, 0.5),
                    center_y=tfd.Normal(0, 0.5),
                    Ie=tfd.LogNormal(jnp.log(150.0), 0.9),
                )
            )
        ]
    )

    prior = tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior]
    )
    return prior