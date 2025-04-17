// Function to generate a shooting star
function createShootingStar() {
    const star = document.createElement('div');
    star.classList.add('shooting-star');

    // Random size
    const size = Math.random() * 2 + 1; // 1px to 3px
    star.style.width = `${size}px`;
    star.style.height = `${size}px`;

    // Random start position
    const startX = Math.random() * window.innerWidth;
    const startY = Math.random() * window.innerHeight / 2;

    // Animation duration
    const duration = Math.random() * 3 + 2;

    // Random angle
    const angle = Math.random() * (Math.PI / 4) - Math.PI / 8;
    const endX = startX + Math.cos(angle) * window.innerWidth;
    const endY = startY + Math.sin(angle) * window.innerHeight;

    star.style.top = `${startY}px`;
    star.style.left = `${startX}px`;
    star.style.animationName = 'shootStar';
    star.style.animationDuration = `${duration}s`;
    star.style.animationTimingFunction = 'linear';
    star.style.filter = 'blur(0.5px)';

    document.body.appendChild(star);

    star.addEventListener('animationend', () => {
        star.remove();
    });
}

// Function to generate multiple shooting stars
function generateShootingStars() {
    setInterval(() => {
        const delay = Math.random() * 1000; // delay between 0 to 1000ms
        setTimeout(createShootingStar, delay);
    }, 700); // Try 700ms for a slightly slower burst
}

// Start generating stars
generateShootingStars();
