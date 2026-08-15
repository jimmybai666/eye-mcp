from eye_mcp.common.config import load_config


def test_load_config():
    """测试配置文件加载功能"""
    config = load_config("config.yaml")
    # 日志配置
    assert config["log_level"] == "INFO"
    assert config["log_dir"] == "./logs"


def test_oct_seg_config():
    """测试 OCT 分割服务配置"""
    config = load_config("config.yaml")
    assert "oct_seg" in config
    assert config["oct_seg"]["device"] == "auto"
    assert config["oct_seg"]["img_size"] == 512
    assert config["oct_seg"]["default_alpha"] == 0.4
    assert "weights_path" in config["oct_seg"]


def test_fovea_location_config():
    """测试 Fovea 定位服务配置"""
    config = load_config("config.yaml")
    assert "fovea_location" in config
    assert config["fovea_location"]["device"] == "auto"
    assert config["fovea_location"]["img_size"] == 512
    assert config["fovea_location"]["line_width"] == 5
    assert "weights_path" in config["fovea_location"]


def test_optic_disc_cup_seg_config():
    """测试视杯视盘分割服务配置"""
    config = load_config("config.yaml")
    assert "optic_disc_cup_seg" in config
    assert config["optic_disc_cup_seg"]["device"] == "auto"
    assert config["optic_disc_cup_seg"]["default_alpha"] == 0.4
    assert "weights_dir" in config["optic_disc_cup_seg"]
    assert config["optic_disc_cup_seg"]["hf_model"] == "pamixsun/segformer_for_optic_disc_cup_segmentation"


def test_vessel_seg_config():
    """测试血管分割服务配置"""
    config = load_config("config.yaml")
    assert "vessel_seg" in config
    assert config["vessel_seg"]["device"] == "auto"
    assert config["vessel_seg"]["default_alpha"] == 0.5
    assert config["vessel_seg"]["patch_size"] == 256
    assert config["vessel_seg"]["stride_height"] == 50
    assert config["vessel_seg"]["stride_width"] == 50
    assert config["vessel_seg"]["batch_size"] == 8
    assert config["vessel_seg"]["target_resolution"] == [1620, 1444]
    assert "weights_path" in config["vessel_seg"]
    assert config["vessel_seg"]["hf_repo"] == "weidai00/RIP-AV-su-lab"
    assert config["vessel_seg"]["hf_filename"] == "G_best.pkl"


def test_lesion_seg_config():
    """测试病灶分割服务配置"""
    config = load_config("config.yaml")
    assert "lesion_seg" in config
    assert config["lesion_seg"]["device"] == "auto"
    assert config["lesion_seg"]["img_size"] == 1024
    assert config["lesion_seg"]["default_alpha"] == 0.5
    assert config["lesion_seg"]["threshold"] == 0.5
    assert "weights_dir" in config["lesion_seg"]
