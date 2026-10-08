import math

import torch

from gpucat.geometry import boxes_overlap, route_coords, segments_touch_box

f = torch.float64


def t(x):
    return torch.tensor(x, dtype=f)


def test_route_coords_lateral_and_arc_length():
    route = t([[[0, 0], [10, 0], [10, 10], [0, 0]]])  # an L, the last point padding (route_n = 3)
    lat, s = route_coords(t([[12.0, 5.0]]), route, torch.tensor([3]))
    assert math.isclose(float(lat), 2.0) and math.isclose(float(s), 15.0)
    lat, s = route_coords(t([[4.0, -3.0]]), route, torch.tensor([3]))
    assert math.isclose(float(lat), 3.0) and math.isclose(float(s), 4.0)


def test_segment_touches_box_only_when_it_reaches_the_footprint():
    c, yaw, l, w = t([[0.0, 0.0]] * 4), t([0.0, 0.0, math.pi / 2, 0.0]), t([4.0] * 4), t([2.0] * 4)
    segs = t([[[-5, 1.5, 5, 1.5]],   # 1.5 m beside a 2 m wide box: clear
              [[-5, 0.9, 5, 0.9]],   # crosses its side
              [[-5, 1.5, 5, 1.5]],   # the box turned 90 degrees: 2 m half-length reaches y = 1.5
              [[0.5, 0.2, 0.8, 0.3]]])  # wholly inside
    assert segments_touch_box(c, yaw, l, w, segs, torch.tensor([1, 1, 1, 1])).tolist() == [False, True, True, True]
    assert segments_touch_box(c[:1], yaw[:1], l[:1], w[:1], t([[[-5, 0.9, 5, 0.9]]]), torch.tensor([0])).tolist() == [False]


def test_boxes_overlap_by_separating_axes():
    c, yaw, size = t([[0.0, 0.0]]), t([0.0]), t([[4.0, 2.0]])
    oc = t([[[4.1, 0.0], [3.9, 0.0], [2.0, 2.3], [2.5, 2.5]]])
    oyaw = t([[0.0, 0.0, 0.0, math.pi / 4]])
    osize = t([[[4.0, 2.0]] * 4])
    # end to end 0.1 m apart; 0.1 m overlap; 0.3 m beside; a rotated box whose corner reaches in
    assert boxes_overlap(c, yaw, size, oc, oyaw, osize)[0].tolist() == [False, True, False, True]
