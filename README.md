# Differential Physics

There are two bodies in a 3D space. Each body is characterized by its mass ($m_a$ and $m_b$) and inertia tensor ($I_a$ and $I_b$). Also there are relative vectors that are pointing from the center of mass and are oriented with the body all the time $r_a$ and $r_b$ that represent the position of a point on each body.
The bodies have initial position $x_a^0$ and $x_b^0$, and initial orientation $q_a^0$ and $q_b^0$. The goal is to determine how to translate and rotate the bodies so that specific points on each body, $r_a$ and $r_b$, align as closely as possible in world coordinates.

**Objective:** Minimize the distance between the two points in world coordinates. This means that the distance between the two points should be as close to zero as possible.

**Constraints:** Ensure that the force needed to bring the two points together is balanced. This means that the corrections applied to each body should be equal in magnitude but opposite in direction. The corrections should consider both the mass and inertia of each body, so the adjustments respect each body's resistance to translation and rotation.

----------------------------

In a 3D space, we have two bodies, each defined by its mass ($m_a$ and $m_b$) and inertia tensor ($I_a$ and $I_b$). Each body has a point of interest represented by vectors $r_a$ and $r_b$, which are fixed relative to the center of mass and rotate with the body. The initial positions of the bodies are $x_a^0$ and $x_b^0$, and their initial orientations are $q_a^0$ and $q_b^0$.

**Goal:** Determine how to translate ($x_a^1$ and $x_b^1$) and rotate ($q_a^1$ and $q_b^1$) the bodies so that the points $r_a$ and $r_b$ align as closely as possible in world coordinates.

**Objective:** Minimize the distance between $r_a$ and $r_b$ in world coordinates, aiming to bring this distance as close to zero as possible.

**Constraints:** Ensure balanced force adjustments by applying equal-magnitude, opposite-direction corrections to each body. These corrections should account for each body's mass and inertia, ensuring that the applied adjustments respect each body’s resistance to movement and rotation.

