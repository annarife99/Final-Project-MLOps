import hydra

@hydra.main(config_path='config', config_name='path.yaml')
def my_app(cfg):
    # Modify the config object
    print(cfg.project_path)

    # Save the config to a file
    # hydra.save(cfg, "config.yaml")

if __name__ == "__main__":
    my_app()