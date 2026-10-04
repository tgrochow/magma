# MAGMA

## Build

### Nix Flake

The `flake.nix` provides a development environment. You can either activate it
manually with `nix develop` or automatically with `direnv` after running `direnv
allow`.

### Dependencies

- [Vulkano](https://github.com/vulkano-rs/vulkano#linux-specific-setup)
- [CMake](https://cmake.org/)
- [Ninja](https://ninja-build.org/)

```sh
pacman -Sy base-devel git python cmake vulkan-devel --noconfirm
```

- [Vulkan driver](https://wiki.archlinux.org/title/Vulkan)

```sh
pacman -S vulkan-radeon vulkan-intel vulkan-nouveau 
```

### Run

```sh
cargo run
```
