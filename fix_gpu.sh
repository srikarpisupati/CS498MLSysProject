#!/bin/bash

echo "======================================================================"
echo "GPU Fix Diagnostic and Auto-Fix Script"
echo "======================================================================"
echo ""

DRIVER_LOADED=false
DRIVER_INSTALLED=false
NOUVEAU_LOADED=false
FIXES_APPLIED=false

echo "1. Checking GPU hardware..."
if lspci 2>/dev/null | grep -qi nvidia; then
    echo "   GPU hardware detected:"
    lspci 2>/dev/null | grep -i nvidia || true
    GPU_DETECTED=true
else
    echo "   ERROR: No NVIDIA GPU hardware found"
    echo "   Exiting - no GPU hardware detected"
    exit 1
fi
echo ""

echo "2. Checking NVIDIA driver kernel module..."
if [ -f /proc/driver/nvidia/version ]; then
    echo "   NVIDIA driver kernel module is loaded:"
    cat /proc/driver/nvidia/version 2>/dev/null || echo "   (Could not read version)"
    DRIVER_LOADED=true
else
    echo "   ERROR: NVIDIA driver kernel module NOT loaded"
    DRIVER_LOADED=false
fi
echo ""

if dpkg -l 2>/dev/null | grep -q "nvidia-driver-[0-9]"; then
    DRIVER_INSTALLED=true
    echo "   NVIDIA driver package is installed"
else
    DRIVER_INSTALLED=false
    echo "   ERROR: NVIDIA driver package is NOT installed"
fi
echo ""

echo "3. Checking for nouveau driver..."
if lsmod | grep -q "^nouveau"; then
    echo "   WARNING: nouveau loaded (interferes with NVIDIA)"
    NOUVEAU_LOADED=true
else
    echo "   nouveau driver is not loaded"
    NOUVEAU_LOADED=false
fi
echo ""

echo "4. Checking nvidia-smi availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "   nvidia-smi is available"
    if nvidia-smi &>/dev/null; then
        echo "   nvidia-smi works correctly"
        echo "   Running nvidia-smi..."
        nvidia-smi
    else
        echo "   WARNING: nvidia-smi can't communicate with driver"
    fi
elif [ -f /usr/bin/nvidia-smi ] || [ -f /usr/local/bin/nvidia-smi ]; then
    echo "   WARNING: nvidia-smi exists but not in PATH"
else
    echo "   ERROR: nvidia-smi not found"
fi
echo ""

echo "5. Checking for module system (common on HPC clusters)..."
if command -v module &> /dev/null; then
    echo "   Module system detected"
    echo ""
    echo "   Available CUDA/NVIDIA modules:"
    if module avail 2>&1 | grep -qi -E "cuda|nvidia|gpu"; then
        module avail 2>&1 | grep -i -E "cuda|nvidia|gpu" | head -10 || echo "     (none found)"
    else
        echo "     (none found)"
    fi
    echo ""
    echo "   Try: module load cuda"
    HAS_MODULE_SYSTEM=true
else
    echo "   WARNING: No module system"
    HAS_MODULE_SYSTEM=false
fi
echo ""

echo "6. Checking conda environment CUDA setup..."
PYTHON_FOUND=false

if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
    echo "   Found conda environment: $CONDA_PREFIX"
    PYTHON_FOUND=true
elif [ -f ~/miniconda3/envs/ml-benchmark/bin/python ]; then
    PYTHON_CMD=~/miniconda3/envs/ml-benchmark/bin/python
    PYTHON_FOUND=true
elif [ -f ~/anaconda3/envs/ml-benchmark/bin/python ]; then
    PYTHON_CMD=~/anaconda3/envs/ml-benchmark/bin/python
    PYTHON_FOUND=true
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
    PYTHON_FOUND=true
    echo "   WARNING: Using system python"
fi

if [ "$PYTHON_FOUND" = true ]; then
    echo "   Checking PyTorch CUDA support..."
    if $PYTHON_CMD -c "import torch" 2>/dev/null; then
        $PYTHON_CMD -c "
import torch
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA compiled: {torch.version.cuda if torch.version.cuda else \"None\"}')
device_count = torch.cuda.device_count()
print(f'   CUDA devices detected: {device_count}')
if device_count > 0:
    print(f'   CUDA available: {torch.cuda.is_available()}')
    for i in range(device_count):
        try:
            print(f'     Device {i}: {torch.cuda.get_device_name(i)}')
        except:
            print(f'     Device {i}: (name unavailable)')
" 2>/dev/null || echo "   (Error checking PyTorch CUDA)"
    else
        echo "   WARNING: PyTorch not installed in this environment"
    fi
else
    echo "   WARNING: Python not found in common locations"
fi
echo ""

echo "7. Checking GPU device permissions..."
if ls /dev/nvidia* 2>/dev/null | head -1 > /dev/null; then
    echo "   GPU devices found:"
    ls -l /dev/nvidia* 2>/dev/null | head -5 || echo "   (Could not list devices)"
    echo ""
    if [ -c /dev/nvidia0 ]; then
        if [ -r /dev/nvidia0 ] && [ -w /dev/nvidia0 ]; then
            echo "   User has read/write access to /dev/nvidia0"
        else
            echo "   WARNING: No permissions for /dev/nvidia0"
            echo "   Current permissions:"
            ls -l /dev/nvidia0 2>/dev/null || true
        fi
    fi
else
    echo "   WARNING: No /dev/nvidia* devices found"
    echo "   (This is normal if driver is not loaded)"
fi
echo ""

NEEDS_FIX=false

if [ "$DRIVER_LOADED" = false ]; then
    NEEDS_FIX=true
fi

if [ "$NOUVEAU_LOADED" = true ]; then
    NEEDS_FIX=true
fi

if [ "$NEEDS_FIX" = true ] && [ "$HAS_MODULE_SYSTEM" = false ]; then
    echo "======================================================================"
    echo "AUTO-FIXING GPU ISSUES"
    echo "======================================================================"
    echo ""
    
    if ! sudo -n true 2>/dev/null; then
        echo "WARNING: Needs sudo access"
        echo "   Run: sudo bash $0"
        exit 1
    fi
    
    if [ "$DRIVER_INSTALLED" = false ]; then
        echo "Step 1: Installing NVIDIA driver..."
        sudo apt update -qq
        if apt-cache policy nvidia-driver-535 &>/dev/null; then
            echo "   Installing nvidia-driver-535..."
            sudo apt install -y nvidia-driver-535 nvidia-utils-535 2>&1 | grep -E "(Setting up|Unpacking|Installing)" || true
            DRIVER_INSTALLED=true
            FIXES_APPLIED=true
        else
            echo "   WARNING: Trying ubuntu-drivers..."
            if command -v ubuntu-drivers &>/dev/null; then
                sudo ubuntu-drivers autoinstall 2>&1 | tail -5 || true
                DRIVER_INSTALLED=true
                FIXES_APPLIED=true
            else
                echo "   ERROR: Auto-install failed"
                echo "   Install: sudo apt install nvidia-driver-535"
            fi
        fi
        echo ""
    else
        echo "Step 1: Driver already installed, skipping..."
        echo ""
    fi
    
    if [ ! -f /etc/modprobe.d/blacklist-nouveau.conf ] || ! grep -q "blacklist nouveau" /etc/modprobe.d/blacklist-nouveau.conf 2>/dev/null; then
        echo "Step 2: Blacklisting nouveau driver..."
        echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf > /dev/null
        echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf > /dev/null
        sudo update-initramfs -u -k all -q 2>&1 | tail -3 || true
        FIXES_APPLIED=true
        echo "   nouveau blacklisted"
        echo ""
    else
        echo "Step 2: nouveau already blacklisted, skipping..."
        echo ""
    fi
    
    if [ "$NOUVEAU_LOADED" = true ]; then
        echo "Step 3: Unloading nouveau driver..."
        sudo rmmod nouveau 2>/dev/null || true
        sudo rmmod drm_kms_helper i2c_algo_bit drm 2>/dev/null || true
        echo "   Attempted to unload nouveau"
        echo ""
    fi
    
    if [ "$DRIVER_INSTALLED" = true ] && [ "$DRIVER_LOADED" = false ]; then
        echo "Step 4: Loading NVIDIA kernel modules..."
        sudo modprobe nvidia 2>&1 || echo "   WARNING: Load failed (may need reboot)"
        sudo modprobe nvidia_uvm 2>&1 || echo "   WARNING: nvidia_uvm load failed"
        sleep 2
        if [ -f /proc/driver/nvidia/version ]; then
            echo "   NVIDIA driver loaded successfully!"
            DRIVER_LOADED=true
            FIXES_APPLIED=true
        else
            echo "   WARNING: Driver installed but not loaded (reboot needed)"
        fi
        echo ""
    fi
    
    if [ "$DRIVER_LOADED" = true ]; then
        echo "Step 5: Verifying GPU access..."
        if command -v nvidia-smi &>/dev/null; then
            if nvidia-smi &>/dev/null; then
                echo "   nvidia-smi works correctly!"
                nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
            else
                echo "   WARNING: nvidia-smi can't communicate with driver"
            fi
        fi
        echo ""
    fi
    
    if [ "$FIXES_APPLIED" = true ]; then
        echo "======================================================================"
        echo "FIXES APPLIED"
        echo "======================================================================"
        echo ""
        if [ "$DRIVER_LOADED" = false ]; then
            echo "WARNING: Reboot required: sudo reboot"
            echo ""
        else
            echo "GPU setup complete! The driver is loaded and should be working."
            echo ""
        fi
    fi
fi

echo "======================================================================"
echo "FINAL VERIFICATION"
echo "======================================================================"
echo ""

if [ -f /proc/driver/nvidia/version ]; then
    echo "NVIDIA driver is loaded"
    cat /proc/driver/nvidia/version | head -1
else
    echo "ERROR: NVIDIA driver is NOT loaded"
    if [ "$DRIVER_INSTALLED" = true ]; then
        echo "   Try: sudo modprobe nvidia nvidia_uvm"
        echo "   Or: sudo reboot"
    fi
fi
echo ""

if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi &>/dev/null; then
        echo "nvidia-smi works correctly"
        echo ""
        echo "GPU Information:"
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
    else
        echo "ERROR: nvidia-smi cannot communicate with driver"
    fi
else
    echo "WARNING: nvidia-smi not found"
fi
echo ""

if [ "$PYTHON_FOUND" = true ]; then
    echo "Testing PyTorch GPU access..."
    if $PYTHON_CMD -c "import torch" 2>/dev/null; then
        $PYTHON_CMD -c "
import torch
device_count = torch.cuda.device_count()
if device_count > 0:
    print(f'PyTorch can see {device_count} GPU(s)')
    try:
        x = torch.randn(3, 3).cuda()
        print(f'Successfully created tensor on GPU: {x.device}')
        print('GPU computation works!')
    except Exception as e:
        print(f'WARNING: GPU detected but computation failed: {e}')
else:
    print('ERROR: PyTorch cannot see GPUs')
    print('   (Try again if driver just loaded)')
" 2>&1 | grep -v "UserWarning" || echo "   (Error testing PyTorch)"
    fi
fi

echo ""
echo "======================================================================"
echo "SUMMARY"
echo "======================================================================"
echo ""

if [ -f /proc/driver/nvidia/version ] && command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "GPU is set up and working!"
    echo ""
    echo "Run: python run_benchmark.py"
else
    echo "WARNING: GPU setup incomplete"
    echo ""
    if [ "$DRIVER_INSTALLED" = false ]; then
        echo "  - Driver needs to be installed"
    fi
    if [ "$DRIVER_LOADED" = false ] && [ "$DRIVER_INSTALLED" = true ]; then
        echo "  - Driver not loaded (reboot: sudo reboot)"
    fi
fi

echo "======================================================================"