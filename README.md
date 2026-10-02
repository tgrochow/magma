# MAGMA

## Build

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
