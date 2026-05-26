# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from io import BytesIO
from pathlib import Path
from typing import Optional

import fire
from termcolor import cprint

from models.datatypes import RawMediaItem
from models.llama3.generation import Llama3

import os
import torch


THIS_DIR = Path(__file__).parent


def get_device():
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    return "cpu"


def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 4,
    world_size: Optional[int] = None,
    quantization_mode: Optional[str] = None,
):
    generator = Llama3.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
        device=get_device(),
    )

    interleaved_contents2 = [
        "The following are multiple choice questions (with answers) about  astronomy.\n\nYou are pushing a truck along a road. Would it be easier to accelerate this truck on Mars? Why? (Assume there is no friction)\nA. It would be harder since the truck is heavier on Mars.\nB. It would be easier since the truck is lighter on Mars.\nC. It would be harder since the truck is lighter on Mars.\nD. It would be the same no matter where you are.\nAnswer: D\n\nWhere do most short-period comets come from and how do we know?\nA. The Kuiper belt; short period comets tend to be in the plane of the solar system just like the Kuiper belt.\nB. The Kuiper belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the Kuiper belt.\nC. The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.\nD. The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.\nAnswer: A\n\nSay the pupil of your eye has a diameter of 5 mm and you have a telescope with an aperture of 50 cm. How much more light can the telescope gather than your eye?\nA. 10000 times more\nB. 100 times more\nC. 1000 times more\nD. 10 times more\nAnswer: A\n\nWhy isn\'t there a planet where the asteroid belt is located?\nA. A planet once formed here but it was broken apart by a catastrophic collision.\nB. There was not enough material in this part of the solar nebula to form a planet.\nC. There was too much rocky material to form a terrestrial planet but not enough gaseous material to form a jovian planet.\nD. Resonance with Jupiter prevented material from collecting together to form a planet.\nAnswer: D\n\nWhy is Mars red?\nA. Because the surface is covered with heavily oxidized (\"rusted\") minerals.\nB. Because the atmosphere scatters more light at bluer wavelengths transmitting mostly red light.\nC. Because Mars is covered with ancient lava flows which are red in color.\nD. Because flowing water on Mars\'s surface altered the surface minerals several billion years ago.\nAnswer: A\n\nWhat is true for a type-Ia (\"type one-a\") supernova?\nA. This type occurs in binary systems.\nB. This type occurs in young galaxies.\nC. This type produces gamma-ray bursts.\nD. This type produces high amounts of X-rays.\nAnswer:",
        "The following are multiple choice questions (with answers) about  clinical knowledge.\n\nThe energy for all forms of muscle contraction is provided by:\nA. ATP.\nB. ADP.\nC. phosphocreatine.\nD. oxidative phosphorylation.\nAnswer: A\n\nWhat is the difference between a male and a female catheter?\nA. Male and female catheters are different colours.\nB. Male catheters are longer than female catheters.\nC. Male catheters are bigger than female catheters.\nD. Female catheters are longer than male catheters.\nAnswer: B\n\nIn the assessment of the hand function which of the following is true?\nA. Abduction of the thumb is supplied by spinal root T2\nB. Opposition of the thumb by opponens policis is supplied by spinal root T1\nC. Finger adduction is supplied by the median nerve\nD. Finger abduction is mediated by the palmar interossei\nAnswer: B\n\nHow many attempts should you make to cannulate a patient before passing the job on to a senior colleague, according to the medical knowledge of 2020?\nA. 4\nB. 3\nC. 2\nD. 1\nAnswer: C\n\nGlycolysis is the name given to the pathway involving the conversion of:\nA. glycogen to glucose-1-phosphate.\nB. glycogen or glucose to fructose.\nC. glycogen or glucose to pyruvate or lactate.\nD. glycogen or glucose to pyruvate or acetyl CoA.\nAnswer: C\n\nWhat size of cannula would you use in a patient who needed a rapid blood transfusion (as of 2020 medical knowledge)?\nA. 18 gauge.\nB. 20 gauge.\nC. 22 gauge.\nD. 24 gauge.\nAnswer:",
    ]

    interleaved_contents = [
        "The color of the sky is blue but sometimes it can also be",
        """\
apple is pomme,
bannana is banane,
cherry is""",
        "1, 2, 3, 5, 8, 13",
        "ba ba black sheep, have you any wool?",
    ]
    if generator.args.vision_chunk_size > 0:
        with open(THIS_DIR / "../../resources/dog.jpg", "rb") as f:
            img = f.read()

        interleaved_contents.append(
            [
                RawMediaItem(type="image", data=BytesIO(img)),
                "If I had to write a haiku for this one",
            ]
        )

    for content in interleaved_contents2:
        cprint(f"{content}", end="")
        batch = [content]
        for token_results in generator.completion(
            batch,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=1
        ):
            result = token_results[0]
            if result.finished:
                break

            cprint(result.text, color="yellow", end="")
        print("\n==================================\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
