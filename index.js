/*
 * Todo (not in chronological order): organize/refactor code and images, comment code
 * *fill in content, responsiveness, make things not shift on reload?, fetch weather
 * button hover sweep right to left?, *footer
 * 
 * Known bugs: when menu changes url reload doesnt reset 
 * 
 * Todo: 
 *  0. Footer (however make sure it doesn't interfere with scroll indicator)
 *  3. host site on github
 */



// About Me Carousel
const carouselTrack = document.querySelector('.about-content-carousel .carousel-track');
const carouselSlides = Array.from(carouselTrack.children);

const carouselLeftBtn = document.querySelector('.about-content-carousel .carousel-button--left');
const carouselRightBtn = document.querySelector('.about-content-carousel .carousel-button--right');

const carouselNav = document.querySelector('.about-content .carousel-nav');
const carouselNavDots = Array.from(carouselNav.children);

carouselLeftBtn.addEventListener('click', prevSlide);
carouselLeftBtn.addEventListener('click', pauseCarouselInterval);
carouselRightBtn.addEventListener('click', nextSlide);
carouselRightBtn.addEventListener('click', pauseCarouselInterval);
carouselNavDots.forEach((elt) => {
    elt.addEventListener('click', skipToSlide);
    elt.index = carouselNavDots.indexOf(elt)
    elt.addEventListener('click', pauseCarouselInterval);
});

function prevSlide() { stepSlide(-1) }
function nextSlide() { stepSlide(1) }
function skipToSlide(evt) { goToSlide(evt.currentTarget.index); }

function modifyCurrentSlide() {
    const currentSlide = carouselTrack.querySelector('img.current-slide');
    let currentSlideIndex;
    if (currentSlide != null) {
        currentSlideIndex = carouselSlides.indexOf(currentSlide);

        // unset current slide
        currentSlide.classList.remove('current-slide');
        carouselNavDots[currentSlideIndex].classList.remove('current-carousel-slide-nav');
    }
    return currentSlideIndex;
}

function stepSlide(direction) {
    // index of current slide ±1
    let newSlideIndex = modifyCurrentSlide() + direction;

    // wrap step around 
    if (newSlideIndex < 0) newSlideIndex = carouselSlides.length - 1;
    if (newSlideIndex >= carouselSlides.length) newSlideIndex = 0;

    goToSlide(newSlideIndex);
}

function goToSlide(index) {
    modifyCurrentSlide();
    const newCurrSlide = carouselSlides[index];

    const slideWidth = newCurrSlide.getBoundingClientRect().width;
    const scrollAmount = slideWidth * index;
    carouselTrack.scroll({
        left: scrollAmount,
        behavior: 'smooth'
    });

    // set new current slide
    newCurrSlide.classList.add('current-slide');
    carouselNavDots[index].classList.add('current-carousel-slide-nav');
}

// continuously slides carousel
let intervalID = window.setInterval(nextSlide, 5000);
function pauseCarouselInterval() {
    window.clearInterval(intervalID);

    // start new interval after delay
    setTimeout(intervalID = window.setInterval(nextSlide, 5000), 5000);
}

// so interval doesn't interrupt navigation
const navbar = document.querySelector('nav.navbar .navbar-nav');
const navItems = Array.from(navbar.children)
navItems.forEach((elt) => {
    elt.children[0].addEventListener('click', pauseCarouselInterval);
});



// Navbar Scroll Indicator
const navBarHeight = 30;
const scrollIndicatorElt = document.querySelector('nav.navbar div.scroll-indicator');
window.addEventListener('scroll', () => {
    const maxHeight = window.document.body.scrollHeight - window.innerHeight; // height of entire webpage - height of viewable portion
    const positionRatio = window.scrollY / maxHeight;
    scrollIndicatorElt.style.top = (navBarHeight * positionRatio) + 'rem';
});



// Contact Me Form
const sendMessageBtn = document.querySelector('div.contact-content input[type="submit"]');
sendMessageBtn.addEventListener('click', () => {
    const subject = document.getElementById('contact-subject-input').value;
    const body = document.getElementById('contact-body-input').value;
    window.open('mailto:charlescohanlon.inquiries@gmail.com?subject=' + subject + '&body=' + body);
});

