from vit_prisma.sae.config import VisionModelSAERunnerConfig


def test_load_VisionModelSAERunnerConfig():

    # Create a configuration instance
    config = VisionModelSAERunnerConfig()

    # Save the configuration to a file
    config.save_config("config.json")

    # Load the configuration from the file
    loaded_config = VisionModelSAERunnerConfig.load_config("config.json")

    # Verify that the loaded configuration matches the original
    assert config == loaded_config

    # Optionally, print the configuration
    config.pretty_print()
