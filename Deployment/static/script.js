document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const chooseImageBtn = document.getElementById('choose-image');
    const canvas = document.getElementById('canvas');
    const doneButton = document.getElementById('done-button');
    const message = document.getElementById('message');
    const ctx = canvas.getContext('2d');
    let img = null;
    let isDrawing = false;
    let startX, startY, rectWidth, rectHeight;

    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
    });

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        loadImage(file);
    });

    chooseImageBtn.addEventListener('click', () => {
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'image/*';
        fileInput.onchange = (e) => {
            const file = e.target.files[0];
            loadImage(file);
        };
        fileInput.click();
    });

    function loadImage(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                dropArea.style.display = 'none';
                canvas.style.display = 'block';
                doneButton.style.display = 'block';
                message.style.display = 'none';
                canvas.addEventListener('mousedown', handleMouseDown);
                canvas.addEventListener('mousemove', handleMouseMove);
                canvas.addEventListener('mouseup', handleMouseUp);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    doneButton.addEventListener('click', () => {
        if (rectWidth !== 0 && rectHeight !== 0) {
            const relativeX = startX;
            const relativeY = startY;
            const relativeWidth = rectWidth;
            const relativeHeight = rectHeight;
            alert(`Rectangle Coordinates: X=${relativeX}, Y=${relativeY}, Width=${relativeWidth}, Height=${relativeHeight}`);
        } else {
            message.style.display = 'block';
        }
    });

    function handleMouseDown(e) {
        isDrawing = true;
        startX = e.offsetX;
        startY = e.offsetY;
        rectWidth = 0;
        rectHeight = 0;
    }

    function handleMouseMove(e) {
        if (!isDrawing) return;

        rectWidth = e.offsetX - startX;
        rectHeight = e.offsetY - startY;

        drawCanvas();
    }

    function handleMouseUp() {
        isDrawing = false;
    }

    function drawCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);

        if (isDrawing) {
            ctx.strokeStyle = 'yellow';
            ctx.lineWidth = 3;
            ctx.strokeRect(startX, startY, rectWidth, rectHeight);
        }
    }
});
