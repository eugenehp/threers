//! Surface properties and collision filtering.

/// How two surfaces' coefficients combine into the one the solver uses.
///
/// Contacts involve two materials, and there is no universally right way to
/// merge them — a rubber ball on ice should probably take the *minimum*
/// friction, while a brake pad on a disc should take the maximum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CombineRule {
    /// `(a + b) / 2` — the default, and what most engines use.
    #[default]
    Average,
    Min,
    Max,
    /// `a * b` — useful when coefficients are treated as multipliers.
    Multiply,
}

impl CombineRule {
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Average => (a + b) * 0.5,
            Self::Min => a.min(b),
            Self::Max => a.max(b),
            Self::Multiply => a * b,
        }
    }

    /// When two contacting surfaces disagree on the rule, the more emphatic one
    /// wins, so a surface that deliberately suppresses or amplifies is not
    /// silently averaged away. `Min` outranks `Max`: a surface asking to damp
    /// everything (a crash mat, mud) should win over one asking to amplify.
    fn priority(self) -> u8 {
        match self {
            Self::Average => 0,
            Self::Multiply => 1,
            Self::Max => 2,
            Self::Min => 3,
        }
    }

    fn stricter(self, other: Self) -> Self {
        if other.priority() > self.priority() {
            other
        } else {
            self
        }
    }
}

/// Friction and bounciness of a surface.
///
/// ```
/// use threers_physics::prelude::*;
///
/// let ice = PhysicsMaterial::new(0.02, 0.0);
/// let rubber = PhysicsMaterial::new(0.9, 0.8);
/// // Averaged by default.
/// assert!((ice.combine(&rubber).friction - 0.46).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsMaterial {
    /// Coulomb friction coefficient. `0` slides freely; `1` is roughly
    /// rubber-on-concrete. Values above `1` are legal.
    pub friction: f32,
    /// Bounciness in `0..=1`. `0` is a dead stop, `1` is a lossless bounce.
    pub restitution: f32,
    pub friction_combine: CombineRule,
    pub restitution_combine: CombineRule,
}

impl Default for PhysicsMaterial {
    /// Moderately grippy and non-bouncy — sane for most props.
    fn default() -> Self {
        Self {
            friction: 0.5,
            restitution: 0.0,
            // Friction is genuinely a property of the *pair* — rubber on ice is
            // slippery — so averaging is the intuitive default.
            friction_combine: CombineRule::Average,
            // Restitution is not. People set it on the thing they care about
            // (the ball) and leave the floor alone; averaging would then halve
            // it, and `restitution = 1.0` would still only bounce at 0.5. Taking
            // the maximum makes a bouncy object bounce as authored, on any
            // surface. Set `restitution_combine` explicitly to opt out.
            restitution_combine: CombineRule::Max,
        }
    }
}

impl PhysicsMaterial {
    pub fn new(friction: f32, restitution: f32) -> Self {
        Self {
            friction: friction.max(0.0),
            restitution: restitution.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Slippery and dead: `friction = 0`, `restitution = 0`.
    pub fn frictionless() -> Self {
        Self::new(0.0, 0.0)
    }

    /// A bouncy ball: `friction = 0.4`, `restitution = 0.85`.
    pub fn bouncy() -> Self {
        Self::new(0.4, 0.85)
    }

    pub fn with_friction_combine(mut self, rule: CombineRule) -> Self {
        self.friction_combine = rule;
        self
    }

    pub fn with_restitution_combine(mut self, rule: CombineRule) -> Self {
        self.restitution_combine = rule;
        self
    }

    /// Merge two contacting surfaces into the effective contact material.
    pub fn combine(&self, other: &Self) -> Self {
        let f_rule = self.friction_combine.stricter(other.friction_combine);
        let r_rule = self.restitution_combine.stricter(other.restitution_combine);
        Self {
            friction: f_rule.apply(self.friction, other.friction),
            restitution: r_rule.apply(self.restitution, other.restitution),
            friction_combine: f_rule,
            restitution_combine: r_rule,
        }
    }
}

/// Bitmask collision filtering, in the style of three.js `Layers`.
///
/// A pair interacts only if **each** collider is a member of a group the other
/// one filters for. That mutual test means "the player ignores pickups" cannot
/// be expressed by one side alone — both must agree — which avoids the classic
/// one-sided-filter bug.
///
/// ```
/// use threers_physics::prelude::*;
///
/// const WORLD: u32 = 1 << 0;
/// const PLAYER: u32 = 1 << 1;
/// const GHOST: u32 = 1 << 2;
///
/// let level = InteractionGroups::new(WORLD, PLAYER | GHOST);
/// let player = InteractionGroups::new(PLAYER, WORLD);
/// let ghost = InteractionGroups::new(GHOST, 0); // collides with nothing
///
/// assert!(level.test(&player));
/// assert!(!level.test(&ghost));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InteractionGroups {
    /// Which groups this collider belongs to.
    pub memberships: u32,
    /// Which groups this collider is willing to interact with.
    pub filter: u32,
}

impl Default for InteractionGroups {
    /// In every group, interacts with everything.
    fn default() -> Self {
        Self::ALL
    }
}

impl InteractionGroups {
    pub const ALL: Self = Self {
        memberships: u32::MAX,
        filter: u32::MAX,
    };
    pub const NONE: Self = Self {
        memberships: 0,
        filter: 0,
    };

    pub const fn new(memberships: u32, filter: u32) -> Self {
        Self {
            memberships,
            filter,
        }
    }

    /// Only ever interacts with colliders in the same single group.
    pub const fn group(bit: u32) -> Self {
        Self {
            memberships: bit,
            filter: bit,
        }
    }

    #[inline]
    pub fn test(&self, other: &Self) -> bool {
        (self.memberships & other.filter) != 0 && (other.memberships & self.filter) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_rules_do_what_they_say() {
        assert_eq!(CombineRule::Average.apply(0.2, 0.6), 0.4);
        assert_eq!(CombineRule::Min.apply(0.2, 0.6), 0.2);
        assert_eq!(CombineRule::Max.apply(0.2, 0.6), 0.6);
        assert!((CombineRule::Multiply.apply(0.5, 0.5) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn the_stricter_rule_wins_a_disagreement() {
        let slick = PhysicsMaterial::new(0.0, 0.0).with_friction_combine(CombineRule::Min);
        let grippy = PhysicsMaterial::new(1.0, 0.0);
        // Min beats Average, so a deliberately frictionless surface stays so.
        assert_eq!(slick.combine(&grippy).friction, 0.0);
        assert_eq!(grippy.combine(&slick).friction, 0.0);
    }

    #[test]
    fn restitution_is_clamped_but_friction_is_not() {
        assert_eq!(PhysicsMaterial::new(3.0, 5.0).restitution, 1.0);
        assert_eq!(PhysicsMaterial::new(3.0, -1.0).restitution, 0.0);
        assert_eq!(PhysicsMaterial::new(3.0, 0.5).friction, 3.0);
        assert_eq!(PhysicsMaterial::new(-2.0, 0.5).friction, 0.0);
    }

    #[test]
    fn interaction_groups_require_mutual_consent() {
        let a = InteractionGroups::new(0b01, 0b10);
        let b = InteractionGroups::new(0b10, 0b01);
        assert!(a.test(&b));
        assert!(b.test(&a));

        // b stops filtering for a: neither direction interacts any more.
        let b = InteractionGroups::new(0b10, 0b00);
        assert!(!a.test(&b));
        assert!(!b.test(&a));
    }

    #[test]
    fn defaults_interact_with_everything() {
        assert!(InteractionGroups::default().test(&InteractionGroups::default()));
        assert!(!InteractionGroups::NONE.test(&InteractionGroups::ALL));
        assert!(InteractionGroups::group(0b100).test(&InteractionGroups::group(0b100)));
        assert!(!InteractionGroups::group(0b100).test(&InteractionGroups::group(0b010)));
    }
}
