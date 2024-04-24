window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());

gtag('config', 'UA-168892872-1');

document.addEventListener('DOMContentLoaded', function() {
    // Event listener for the Text button
    document.getElementById('submit-text').addEventListener('click', function(event) {
        event.preventDefault();  // Prevent form submission
        document.getElementById('text-search').style.display = 'block';
        document.getElementById('text-multi-search').style.display = 'none'; // Show the text search form
        document.getElementById('image-search').style.display = 'none'; // Hide the image search form
    });

   
        document.getElementById('submit-multiLingual').addEventListener('click', function(event) {
            event.preventDefault();  // Prevent form submission
            document.getElementById('text-search').style.display = 'none';
             // Show the text search form
             document.getElementById('text-multi-search').style.display = 'block';
            document.getElementById('image-search').style.display = 'none'; // Hide the image search form
        });
 
    // Event listener for the Image button
    document.getElementById('submit-Image').addEventListener('click', function(event) {
        event.preventDefault();  // Prevent form submission
        document.getElementById('text-search').style.display = 'none';
        document.getElementById('text-multi-search').style.display = 'none';  // Hide the text search form
        document.getElementById('image-search').style.display = 'block'; // Show the image search form
    });



document.getElementById('imageInput').addEventListener('change', function(event) {
    var reader = new FileReader();
    reader.onload = function(){
        var output = document.getElementById('preview');
        output.src = reader.result;
        output.style.display = 'block'; // Show the preview
    };
    reader.readAsDataURL(event.target.files[0]);
});
});


