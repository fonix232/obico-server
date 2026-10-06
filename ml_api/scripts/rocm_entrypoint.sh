#!/bin/sh
# Entrypoint of the ROCm image (ml_api/Containerfile.rocm).
#
# MIOpen's assembly Winograd kernel for 3x3 convolutions
# (ConvBinWinogradRxSf3x2, miopenSp3AsmConv_v30_3_1_gfx11_fp32_f3x2) dies with
# "unspecified launch failure" on gfx1103 (Radeon 780M/760M), and the queues it
# leaves behind make the kernel driver reset the whole GPU, taking every other
# GPU user on the host down with it. With the solver off, MIOpen picks another
# one and the detections match ONNX Runtime's. The kernel is one gfx11 code
# object for the whole 11.0 family, so it is switched off on any 11.0.x part
# (gfx1100 to gfx1103) and kept on 11.5 (gfx1150, Radeon 890M) and gfx12. KFD
# reports the target as gfx_target_version with two decimal digits per field:
# 110003 is gfx1103, 110500 is gfx1150.
#
# MIOpen reads the variable from the environment, so it is set here, before
# gunicorn starts, rather than from Python. A value already in the environment
# wins either way.
#
#   rocm_entrypoint.sh COMMAND [ARG...]
#   rocm_entrypoint.sh --self-test

set -eu

KFD_TOPOLOGY="${KFD_TOPOLOGY:-/sys/class/kfd/kfd/topology/nodes}"

# Succeeds when any KFD node under $1 is a gfx11.0.x (gfx1100 to gfx1103).
has_gfx110x() {
    for props in "$1"/*/properties; do
        # Without the GPU passed in (/dev/kfd and its /dev/dri render node),
        # sysfs still lists the nodes but refuses to read them (EPERM): no
        # GPU, so nothing to switch off.
        [ -r "${props}" ] || continue
        if grep -qEx 'gfx_target_version 1100[0-9]{2}' "${props}" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

fail() {
    echo "rocm_entrypoint self-test: $*" >&2
    exit 1
}

self_test() {
    tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' EXIT

    # A CPU node plus one GPU node per host, as KFD lays them out.
    for host in 780m:110003 7900xtx:110000 7800xt:110001 7600:110002 \
        890m:110500 strix-halo:110501 9070xt:120001 gfx1010:100100 lookalike:1100030; do
        name="${host%%:*}"
        mkdir -p "${tmp}/${name}/0" "${tmp}/${name}/1"
        printf 'cpu_cores_count 16\ngfx_target_version 0\n' > "${tmp}/${name}/0/properties"
        printf 'simd_count 24\ngfx_target_version %s\n' "${host#*:}" > "${tmp}/${name}/1/properties"
    done
    mkdir -p "${tmp}/no-kfd"

    for name in 780m 7900xtx 7800xt 7600; do
        has_gfx110x "${tmp}/${name}" || fail "${name} not taken for a gfx11.0.x"
    done
    for name in 890m strix-halo 9070xt gfx1010 lookalike no-kfd; do
        if has_gfx110x "${tmp}/${name}"; then
            fail "${name} taken for a gfx11.0.x"
        fi
    done

    # Through the real dispatch: set on gfx11.0.x, left alone elsewhere, and
    # an explicit value kept.
    unset MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2
    out="$(KFD_TOPOLOGY="${tmp}/780m" sh "$0" env 2>/dev/null)"
    echo "${out}" | grep -qx 'MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0' || fail "not set on gfx1103"
    out="$(KFD_TOPOLOGY="${tmp}/890m" sh "$0" env 2>/dev/null)"
    if echo "${out}" | grep -q '^MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2='; then
        fail "set on gfx1150"
    fi
    out="$(MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=1 KFD_TOPOLOGY="${tmp}/780m" sh "$0" env 2>/dev/null)"
    echo "${out}" | grep -qx 'MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=1' || fail "explicit value overridden"

    echo "rocm_entrypoint self-test: ok"
}

if [ "${1:-}" = --self-test ]; then
    self_test
    exit 0
fi

if [ -z "${MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2+set}" ] && has_gfx110x "${KFD_TOPOLOGY}"; then
    export MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2=0
    echo "rocm_entrypoint: gfx11.0.x GPU found, MIOpen ConvBinWinogradRxSf3x2 disabled" >&2
fi

exec "$@"
