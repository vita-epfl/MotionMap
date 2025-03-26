
let index = 0;
const carousel = document.getElementById('carousel');
const total = carousel.children.length;

function updateSlide() {
  carousel.style.transform = `translateX(-${index * 100}%)`;
}

function prevSlide() {
  index = (index - 1 + total) % total;
  updateSlide();
}

function nextSlide() {
  index = (index + 1) % total;
  updateSlide();
}

function goToSlide(i) {
  index = i;
  updateSlide();
}
