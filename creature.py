# === File: creature.py ===
# Simple 4-leg creature that actually moves.
# Uses genome control params for the 4 hip joints.

from xml.dom.minidom import getDOMImplementation
from enum import Enum
import numpy as np
import pybullet as p
import genome


class MotorType(Enum):
    PULSE = 1
    SINE = 2


class Motor:
    def __init__(self, control_waveform, control_amp, control_freq, phase_offset=0.0):
        self.motor_type = MotorType.PULSE if control_waveform <= 0.5 else MotorType.SINE
        self.amp = float(control_amp)
        self.freq = float(control_freq)
        self.phase = float(phase_offset)

    def get_output(self):
        self.phase = (self.phase + self.freq) % (2 * np.pi)
        if self.motor_type == MotorType.PULSE:
            return self.amp if self.phase < np.pi else -self.amp
        return self.amp * np.sin(self.phase)


def _clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


class Creature:
    """
    This replaces the old "expanding link tree" URDF with a stable walker:
      - 1 body (box)
      - 4 legs (cylinders)
      - 4 hip joints (revolute, pitch axis so legs swing forward/back)
    Genome still controls:
      - leg length/radius (from link_length/link_radius)
      - motor waveform/amp/freq (control_* fields)
      - motor phase offset (from joint_origin_rpy_1)
    """

    def __init__(self, gene_count=4, gene=None):
        self.spec = genome.Genome.get_gene_spec()
        self.dna = gene if gene is not None else genome.Genome.get_random_genome(len(self.spec), gene_count)

        self.motors = None
        self.start_position = None
        self.last_position = None

    def _gene_dicts(self):
        # One dict per gene (we will use first 4 genes as 4 legs)
        gdicts = genome.Genome.get_genome_dicts(self.dna, self.spec)
        return gdicts[:4]

    def to_xml(self):
        g = self._gene_dicts()

        # ---- Body params (kept stable so it’s not a weird blob)
        body_l, body_w, body_h = 0.55, 0.35, 0.14
        body_mass = 2.0

        # ---- Leg positions (front-left, front-right, back-left, back-right)
        hip_positions = [
            (+0.22, +0.16, 0.0),
            (+0.22, -0.16, 0.0),
            (-0.22, +0.16, 0.0),
            (-0.22, -0.16, 0.0),
        ]

        domimpl = getDOMImplementation()
        adom = domimpl.createDocument(None, "robot", None)
        robot_tag = adom.documentElement
        robot_tag.setAttribute("name", "simple_walker")

        # ---- Base link
        base_link = adom.createElement("link")
        base_link.setAttribute("name", "base")

        visual = adom.createElement("visual")
        geom = adom.createElement("geometry")
        box = adom.createElement("box")
        box.setAttribute("size", f"{body_l} {body_w} {body_h}")
        geom.appendChild(box)
        visual.appendChild(geom)
        base_link.appendChild(visual)

        collision = adom.createElement("collision")
        geom2 = adom.createElement("geometry")
        box2 = adom.createElement("box")
        box2.setAttribute("size", f"{body_l} {body_w} {body_h}")
        geom2.appendChild(box2)
        collision.appendChild(geom2)
        base_link.appendChild(collision)

        inertial = adom.createElement("inertial")
        mass = adom.createElement("mass")
        mass.setAttribute("value", str(body_mass))
        inertial.appendChild(mass)
        inertia = adom.createElement("inertia")
        inertia.setAttribute("ixx", "0.1")
        inertia.setAttribute("iyy", "0.1")
        inertia.setAttribute("izz", "0.1")
        inertia.setAttribute("ixy", "0")
        inertia.setAttribute("ixz", "0")
        inertia.setAttribute("iyx", "0")
        inertial.appendChild(inertia)
        base_link.appendChild(inertial)

        robot_tag.appendChild(base_link)

        # ---- Legs + joints
        for i in range(4):
            gd = g[i]

            # Use genome values but clamp to realistic sizes
            leg_len = _clamp(gd.get("link_length", 0.6), 0.18, 0.55)
            leg_rad = _clamp(gd.get("link_radius", 0.12), 0.04, 0.12)
            leg_mass = _clamp(gd.get("link_mass", 0.5), 0.15, 1.0)

            leg_name = f"leg{i}"

            # link
            leg_link = adom.createElement("link")
            leg_link.setAttribute("name", leg_name)

            v = adom.createElement("visual")
            gtag = adom.createElement("geometry")
            cyl = adom.createElement("cylinder")
            cyl.setAttribute("length", str(leg_len))
            cyl.setAttribute("radius", str(leg_rad))
            gtag.appendChild(cyl)
            v.appendChild(gtag)
            leg_link.appendChild(v)

            c = adom.createElement("collision")
            gtag2 = adom.createElement("geometry")
            cyl2 = adom.createElement("cylinder")
            cyl2.setAttribute("length", str(leg_len))
            cyl2.setAttribute("radius", str(leg_rad))
            gtag2.appendChild(cyl2)
            c.appendChild(gtag2)
            leg_link.appendChild(c)

            inert = adom.createElement("inertial")
            mtag = adom.createElement("mass")
            mtag.setAttribute("value", str(leg_mass))
            inert.appendChild(mtag)
            itag = adom.createElement("inertia")
            itag.setAttribute("ixx", "0.02")
            itag.setAttribute("iyy", "0.02")
            itag.setAttribute("izz", "0.02")
            itag.setAttribute("ixy", "0")
            itag.setAttribute("ixz", "0")
            itag.setAttribute("iyx", "0")
            inert.appendChild(itag)
            leg_link.appendChild(inert)

            robot_tag.appendChild(leg_link)

            # joint (hip)
            joint = adom.createElement("joint")
            joint.setAttribute("name", f"hip{i}")
            joint.setAttribute("type", "revolute")

            parent = adom.createElement("parent")
            parent.setAttribute("link", "base")
            joint.appendChild(parent)

            child = adom.createElement("child")
            child.setAttribute("link", leg_name)
            joint.appendChild(child)

            axis = adom.createElement("axis")
            # pitch axis (Y) makes legs swing forward/back
            axis.setAttribute("xyz", "0 1 0")
            joint.appendChild(axis)

            limit = adom.createElement("limit")
            limit.setAttribute("effort", "60")
            limit.setAttribute("velocity", "10")
            limit.setAttribute("lower", str(-0.9))
            limit.setAttribute("upper", str(+0.9))
            joint.appendChild(limit)

            origin = adom.createElement("origin")
            hx, hy, hz = hip_positions[i]
            # put joint at side of body; leg cylinder centered on joint, so drop it down by half-length
            origin.setAttribute("xyz", f"{hx} {hy} {hz}")
            origin.setAttribute("rpy", "0 0 0")
            joint.appendChild(origin)

            robot_tag.appendChild(joint)

        return '<?xml version="1.0"?>' + robot_tag.toprettyxml()

    def get_motors(self):
        if self.motors is None:
            motors = []
            g = self._gene_dicts()

            # Trot-ish offsets help movement (two legs opposite phase)
            default_offsets = [0.0, np.pi, np.pi, 0.0]

            for i in range(4):
                gd = g[i]
                wf = float(gd.get("control_waveform", 1.0))
                amp = _clamp(gd.get("control_amp", 0.6), 0.2, 1.2)
                freq = _clamp(gd.get("control_freq", 0.08), 0.02, 0.18)

                # Use gene if present, otherwise a useful default
                gene_phase = gd.get("joint_origin_rpy_1", 0.0)
                phase = float(gene_phase) if gene_phase != 0.0 else default_offsets[i]

                motors.append(Motor(wf, amp, freq, phase_offset=phase))

            self.motors = motors

        return self.motors

    def control_motors(self, robot_id):
        motors = self.get_motors()
        num_joints = p.getNumJoints(robot_id)

        for i, m in enumerate(motors):
            if i >= num_joints:
                break

            output = float(m.get_output())
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=output,
                force=60.0,
                positionGain=0.25,
                velocityGain=1.0
            )

    def update_position(self, pos):
        if self.start_position is None:
            self.start_position = pos
        else:
            self.last_position = pos

    def get_distance_travelled(self):
        if self.start_position is None or self.last_position is None:
            return 0.0
        p1 = np.asarray(self.start_position)
        p2 = np.asarray(self.last_position)
        return float(np.linalg.norm(p1 - p2))

    def update_dna(self, dna):
        self.dna = dna
        self.motors = None
        self.start_position = None
        self.last_position = None
