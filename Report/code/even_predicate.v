Inductive even : N -> Prop :=
  | even_0 : even 0
  | even_S n : odd n -> even (n + 1)
with odd : N -> Prop :=
  | odd_S n : even n -> odd (n + 1).
