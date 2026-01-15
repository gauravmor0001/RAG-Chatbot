document.addEventListener("DOMContentLoaded", ()=>{
    const fileInput=document.getElementById('fileInput');
    const uploadButton=document.getElementById('uploadbutton');
    const uploadstatus=document.getElementById('uploadstatus');
    const chatwindow=document.getElementById('chatwindow');
    const userInput=document.getElementById('userinput');
    const sendButton=document.getElementById('sendbutton');


// when user clicks upload button we can not just send it in text format we use formdata,it sends data in key-value pair.
    uploadButton.addEventListener('click',async()=>{
        const file=fileInput.files[0]; 
        // .files work as an array we are just taking the first file.
        if(!file){
            alert("Please select a file to upload");
            return;
        }
        //create package of formdata to send to server 
        const formData=new FormData();
        formData.append('file',file); 
        uploadstatus.innerText="Uploading...";

        try{
            // fetch send api request to server
            const response=await fetch('http://127.0.0.1:5000/upload',{
                method:'POST',
                body:formData
            });
            const data = await response.json(); //read server response
            if(data.stats){
                uploadstatus.innerText = `Success! Mode: ${data.stats.mode}`;
            }else{
                uploadstatus.innerText = "Upload complete.";
            }

        }catch (error) {
            console.error('Error:', error);
            uploadstatus.innerText = "Upload failed. Is the server running?";
        }
    });

    // chat logic
    async function sendMessage(){
        const message=userInput.value.trim();
        // trim removes extra spaces from start and end
        if(!message){
            return;
        }
        addMessageToUI("You",message,"user-message");
        userInput.value=""; // clear input field

        try{
            const response = await fetch('http://127.0.0.1:5000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json' // Tells Python this is JSON, not needed to send file
                },
                body: JSON.stringify({ message: message }) // Convert JS object to text string
            });
            const data = await response.json();

            if (data.response) {
                addMessageToUI("Bot", data.response, "bot-message");
            } else {
                addMessageToUI("System", "Error: " + JSON.stringify(data), "error-message");
            }
        } catch (error) {
            addMessageToUI("System", "Server connection failed.", "error-message");
        }
    }
    sendButton.addEventListener('click',sendMessage);
    userInput.addEventListener('keypress',(e)=>{
        if(e.key==='Enter'){
            sendMessage();
        }
    });

    function addMessageToUI(sender,text,className){
        const messageDiv=document.createElement('div');
        messageDiv.className=`message ${className}`;

        const senderSpan=document.createElement('strong');
        senderSpan.textContent=sender+": ";

        const textSpan=document.createElement('span');
        textSpan.textContent=text;

        messageDiv.appendChild(senderSpan);
        messageDiv.appendChild(textSpan);
        
        chatwindow.appendChild(messageDiv);
        
        // Auto-scroll to the bottom so you always see the newest message
        chatwindow.scrollTop = chatwindow.scrollHeight;
    }
});