import argparse
from typing import List
import json

from SkillShape import SkillShape
from TrampScorer import TrampScorer

parser = argparse.ArgumentParser()
parser.add_argument("environment_root", type=str)
parser.add_argument("left_cutoff", type=int)
parser.add_argument("right_cutoff", type=int)
parser.add_argument("bed_y", type=int)

args = parser.parse_args()

scorer = TrampScorer(args.environment_root)
scorer.set_constants(args.left_cutoff, args.right_cutoff, args.bed_y, 40, 4, 20, 300, 300, 0.5)
scorer.allocate_memory()
scorer.start()


class Skill:
    ys: List[int]
    angle: float
    shape: SkillShape
    video_name: str

    def __init__(self):
        self.ys = []


skills: List[Skill] = [Skill()]

current_frame = scorer.batches[0].frames[0]
next_frame = current_frame.next_frame
while next_frame is not None:
    skills[-1].ys.append(current_frame.smoothed_position[1])
    if current_frame.skill_index < next_frame.skill_index:
        skills[-1].angle = scorer.skill_angles[current_frame.skill_index]
        skills[-1].shape = scorer.skill_shapes[current_frame.skill_index]
        skills[-1].video_name = scorer.skill_video_names[current_frame.skill_index]
        skills.append(Skill())

    current_frame = next_frame
    next_frame = current_frame.next_frame

skills[-1].angle = scorer.skill_angles[current_frame.skill_index]
skills[-1].shape = scorer.skill_shapes[current_frame.skill_index]
skills[-1].video_name = scorer.skill_video_names[current_frame.skill_index]

file = open("%s/stats.json" % args.environment_root, "w")
file.write(json.dumps({
    "sourceName": scorer.source_name,
    "skills": [skill.__dict__ for skill in skills]
}).replace("video_name", "videoName"))
file.close()