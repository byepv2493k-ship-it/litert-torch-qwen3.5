import jax
import jax.numpy as jnp
import uuid

def check_collisions():
    # If the UUID is seeded or forked incorrectly, it might repeat.
    u1 = f"{uuid.uuid4().hex[:16]}"
    import multiprocessing
    def get_u():
        return uuid.uuid4().hex[:16]
    with multiprocessing.Pool(2) as p:
        u2, u3 = p.map(get_u, [1, 2])
    print(u1, u2, u3)
check_collisions()
