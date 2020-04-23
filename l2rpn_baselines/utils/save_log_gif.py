# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
from grid2op.Plot import EpisodeReplay


def save_log_gif(path_log, res, gif_name="epidose.gif"):
    """
    Output a gif named (by default "episode.gif") that is the replay of the episode in a gif format,
    for each episode in the input.

    Parameters
    ----------
    path_log: ``str``
        Path where the log of the agents are saved.

    res: ``list``
        List resulting from the call to `runner.run`

    gif_name: ``str``
        Name of the gif that will be used.

    """
    ep_replay = EpisodeReplay(agent_path=path_log)
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        ep_replay.replay_episode(chron_name,
                                 video_name=os.path.join(path_log, chron_name, gif_name),
                                 display=False)