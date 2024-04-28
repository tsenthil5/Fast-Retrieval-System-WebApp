window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());

gtag('config', 'UA-168892872-1');

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('submit-text').addEventListener('click', function(event) {
        event.preventDefault();  
        document.getElementById('text-search').style.display = 'block';
        document.getElementById('text-multi-search').style.display = 'none'; 
        document.getElementById('searchForm').style.display = 'none'; 
        document.getElementById('image-search').style.display = 'none';
    });

    

   
        document.getElementById('submit-multiLingual').addEventListener('click', function(event) {
            event.preventDefault();  
            document.getElementById('text-search').style.display = 'none';
             document.getElementById('text-multi-search').style.display = 'block';
            document.getElementById('searchForm').style.display = 'none';
            document.getElementById('image-search').style.display = 'none';
        });
        document.getElementById('submit-Speech').addEventListener('click', function(event) {
            event.preventDefault();  
            document.getElementById('text-search').style.display = 'none';
             document.getElementById('text-multi-search').style.display = 'none';
            document.getElementById('searchForm').style.display = 'block';
            document.getElementById('image-search').style.display = 'none';
        });

    document.getElementById('submit-Image').addEventListener('click', function(event) {
        event.preventDefault(); 
        document.getElementById('text-search').style.display = 'none';
        document.getElementById('text-multi-search').style.display = 'none';
        document.getElementById('searchForm').style.display = 'none';
        document.getElementById('image-search').style.display = 'block'; 
        
    });



document.getElementById('imageInput').addEventListener('change', function(event) {
    var reader = new FileReader();
    reader.onload = function(){
        var output = document.getElementById('preview');
        output.src = reader.result;
        output.style.display = 'block';
    };
    reader.readAsDataURL(event.target.files[0]);
});
});

document.addEventListener('DOMContentLoaded', function() {
    var searchInput = document.getElementById('searchInput');
    var startBtn = document.getElementById('startBtn');
    var SpeechRecognition = SpeechRecognition || webkitSpeechRecognition;
    if (typeof SpeechRecognition === "undefined") {
        startBtn.disabled = true;
        alert('Speech recognition is not supported in this browser. Please use Google Chrome or a similar browser.');
        return;
    }

    var recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true; 
    recognition.lang = "en-US";
    
    recognition.onresult = function(event) {
        var lastResult = event.results.length - 1;
        console.log(event.results[lastResult][0].transcript)
        searchInput.value = event.results[lastResult][0].transcript;
    };



    recognition.onerror = function(event) {
        console.error("Speech recognition error", event.error);
    };

    startBtn.onclick = function() {
        recognition.start(); 
    };

});
