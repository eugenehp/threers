# A threers game

```sh
cargo run
```

WASD drives the cube. Everything is in `src/main.rs`.

- **Tune while it runs:** `tuning/player.json` appears on the first run. Edit it and save; the game picks the change up.
- **Look inside:** add `.plugin(threers_ecs::dev::DevTools)`. F3 shows the panel; F7 picks an entity, F8 a field, F9 and F10 change it; F12 records your input to `recordings/`.
- **Test it:** `TestGame::new(build)` runs the game with no window, and `Recording::from_text` plays back a recording.

The guide is `docs/ecs.md` in the threers repository.
