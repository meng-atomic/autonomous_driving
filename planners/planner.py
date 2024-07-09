#!/usr/bin/python3

class PlannerInterface:
  def __init__(self):
    pass

  def plan_on_reference_line(self, init_state, target, ref_line):
    raise NotImplementedError