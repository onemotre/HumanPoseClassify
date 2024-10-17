import os 

ALL_PAR_NAME = ["origin", "classification", "extend"]
ALL_ACTIONS = ["stand", "squat", "one-foot-stand", 
               "open-arms", "arms-skimbo"]

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
video_root = os.path.join(os.getcwd(), "assets", "videos")
pic_with_keypoints_root = os.path.join(
    os.getcwd(), "data", "pic_keypoints_label"
)

def iterate_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def create_keypoint_label_data_dir(video_path):
    '''
    create directory for every video
    @param video_path: video path
    '''
    video_name = os.path.basename(video_path)
    if not os.path.exists(pic_with_keypoints_root):
        os.makedirs(pic_with_keypoints_root)
    root_dir = os.path.join(pic_with_keypoints_root, video_name.split(".")[0])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for name in ALL_PAR_NAME:
        parent_dir = os.path.join(root_dir, name)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
    for action in ALL_ACTIONS:
        action_dir = os.path.join(root_dir, ALL_PAR_NAME[1], action)
        if not os.path.exists(action_dir):
            os.makedirs(action_dir)

def get_origin_pic_dir(video_path):
    return os.path.join(
        pic_with_keypoints_root,
        os.path.basename(video_path).split(".")[0],
        "origin"
    )

def get_extend_pic_dir(video_path):
     return os.path.join(
        pic_with_keypoints_root,
        os.path.basename(video_path).split(".")[0],
        "extend"
    )

def get_action_dir(video_path, action):
    '''
    get the target action directory of a video
    @param video_path: the full path of the video
    @param action: the action of the person
    '''
    return os.path.join(
        pic_with_keypoints_root,
        os.path.basename(video_path).split(".")[0],
        f"classification",
        action
    )

def get_origin_data_path(video_path):
    '''
    get the origin data path of a video (after yolo detect)
    @param video_path: the full path of the video
    '''
    return os.path.join(
        pic_with_keypoints_root,
        os.path.basename(video_path).split(".")[0]
    )

def test():
  # 测试
  for file in iterate_files(video_root):
      if file.endswith(".mp4"):
          video_name = os.path.basename(file).split(".")[0]
          create_keypoint_label_data_dir(video_name)

if __name__ == "__main__":
    test()