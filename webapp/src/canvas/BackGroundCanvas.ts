export class MatrixCanvas {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private columns: number = 0;
    private drops: number[] = [];
    private animId: number = 0;
    private chars = 'アイウエオカキクケコサシスセソタチツテトナニヌネノ0123456789ABCDEF<>{}[]|01';

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d')!;
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    private resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.columns = Math.floor(this.canvas.width / 16);
        this.drops = Array(this.columns).fill(1);
    }

    private draw() {
        const ctx = this.ctx;
        const { width, height } = this.canvas;

        // Trail fade
        ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
        ctx.fillRect(0, 0, width, height);

        for (let i = 0; i < this.drops.length; i++) {
            const char = this.chars[Math.floor(Math.random() * this.chars.length)];
            const x = i * 16;
            const y = this.drops[i] * 16;

            // Head glow (brighter)
            if (Math.random() > 0.95) {
                ctx.fillStyle = '#ffffff';
                ctx.shadowColor = '#00ff41';
                ctx.shadowBlur = 8;
            } else {
                // Body gradient: bright green → dim
                const alpha = Math.random() * 0.5 + 0.3;
                ctx.fillStyle = `rgba(0, ${Math.floor(180 + Math.random() * 75)}, 65, ${alpha})`;
                ctx.shadowColor = '#00ff41';
                ctx.shadowBlur = 4;
            }

            ctx.font = `${Math.random() > 0.98 ? 'bold ' : ''}13px Courier New`;
            ctx.fillText(char, x, y);
            ctx.shadowBlur = 0;

            // Reset drop
            if (y > height && Math.random() > 0.975) {
                this.drops[i] = 0;
            }
            this.drops[i]++;
        }
    }

    start() {
        const loop = () => {
            this.draw();
            this.animId = requestAnimationFrame(loop);
        };
        loop();
    }

    stop() {
        cancelAnimationFrame(this.animId);
    }
}