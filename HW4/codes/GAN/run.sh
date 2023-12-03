python3 main.py --do_train --latent_dim 16 --generator_hidden_dim 16 --discriminator_hidden_dim 16
python3 main.py --do_train --latent_dim 16 --generator_hidden_dim 100 --discriminator_hidden_dim 100
python3 main.py --do_train --latent_dim 100 --generator_hidden_dim 16 --discriminator_hidden_dim 16
python3 main.py --do_train --latent_dim 100 --generator_hidden_dim 100 --discriminator_hidden_dim 100

python3 main.py  --latent_dim 100 --generator_hidden_dim 100 --discriminator_hidden_dim 100 --interpolation
python3 main.py  --latent_dim 100 --generator_hidden_dim 100 --discriminator_hidden_dim 100 --extrapolation
python3 main.py  --latent_dim 100 --generator_hidden_dim 100 --discriminator_hidden_dim 100 --test_collapse